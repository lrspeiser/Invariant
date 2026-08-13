"""Sigma-Core-native cheap-to-formal-to-observational evaluation orchestration.

The runner is deliberately in-memory and deterministic. Domain packs own semantic evaluation;
this module owns ordering, binding, fail-closed stopping, and the observational-opening boundary.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .sigma_core import (
    ArtifactRef,
    CandidateArtifact,
    DomainPack,
    DomainPackDescriptor,
    DomainPackRef,
    GateOutcome,
    OutcomeStatus,
    StageOutcome,
    canonical_sha256,
    run_gate,
    run_stage,
)

LADDER_SCHEMA = "sigma-candidate-evaluation-ladder-1.0"
RESULT_SCHEMA = "sigma-candidate-evaluation-result-1.0"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_PHASE_ORDER = {"cheap": 0, "symbolic": 1, "formal": 2, "observational": 3}
_CLAIM_BOUNDARY = {
    "generator_output_treated_as_truth": False,
    "failed_hard_gate_compensated_by_soft_metric": False,
    "observations_opened_before_formal_admission": False,
    "novelty_established": False,
    "promotion_authorized_by_this_runner": False,
}


class EvaluationLadderError(ValueError):
    """A ladder, result, or replay crossed the fail-closed evaluation boundary."""


class EvaluationPhase(str, Enum):
    CHEAP = "cheap"
    SYMBOLIC = "symbolic"
    FORMAL = "formal"
    OBSERVATIONAL = "observational"


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise EvaluationLadderError(f"{label} keys changed")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise EvaluationLadderError(f"{label} is not a canonical identifier")
    return value


@dataclass(frozen=True, slots=True)
class EvaluationStep:
    stage_id: str
    gate_id: str
    phase: EvaluationPhase

    def __post_init__(self) -> None:
        _identifier(self.stage_id, "stage_id")
        _identifier(self.gate_id, "gate_id")
        if not isinstance(self.phase, EvaluationPhase):
            raise EvaluationLadderError("phase must be an EvaluationPhase")

    def to_dict(self) -> dict[str, str]:
        return {
            "stage_id": self.stage_id,
            "gate_id": self.gate_id,
            "phase": self.phase.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationStep:
        _exact_keys(value, {"stage_id", "gate_id", "phase"}, "evaluation step")
        try:
            phase = EvaluationPhase(value["phase"])
        except (TypeError, ValueError) as error:
            raise EvaluationLadderError("unregistered evaluation phase") from error
        return cls(str(value["stage_id"]), str(value["gate_id"]), phase)


@dataclass(frozen=True, slots=True)
class EvaluationLadder:
    domain_pack: DomainPackRef
    steps: tuple[EvaluationStep, ...]
    content_sha256: str
    schema_version: str = LADDER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != LADDER_SCHEMA:
            raise EvaluationLadderError("evaluation ladder schema changed")
        if not self.steps or self.steps[0].phase is not EvaluationPhase.CHEAP:
            raise EvaluationLadderError("evaluation ladder must begin with a cheap phase")
        if len({step.stage_id for step in self.steps}) != len(self.steps) or len(
            {step.gate_id for step in self.steps}
        ) != len(self.steps):
            raise EvaluationLadderError("evaluation ladder stage and gate IDs must be unique")
        orders = tuple(_PHASE_ORDER[step.phase.value] for step in self.steps)
        if orders != tuple(sorted(orders)):
            raise EvaluationLadderError("evaluation phases cannot move backward")
        first_observational = next(
            (
                index
                for index, step in enumerate(self.steps)
                if step.phase is EvaluationPhase.OBSERVATIONAL
            ),
            None,
        )
        if first_observational is not None and not any(
            step.phase is EvaluationPhase.FORMAL for step in self.steps[:first_observational]
        ):
            raise EvaluationLadderError("observational evaluation requires an earlier formal phase")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise EvaluationLadderError("evaluation ladder canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain_pack": self.domain_pack.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def create(
        cls,
        descriptor: DomainPackDescriptor,
        steps: Sequence[EvaluationStep],
    ) -> EvaluationLadder:
        ordered = tuple(steps)
        if tuple(step.stage_id for step in ordered) != tuple(
            stage.stage_id for stage in descriptor.stages
        ):
            raise EvaluationLadderError("ladder must cover every domain stage in descriptor order")
        for index, step in enumerate(ordered):
            gate = descriptor.gate(step.gate_id)
            if gate.to_stage != step.stage_id:
                raise EvaluationLadderError("ladder gate does not admit its declared stage")
            expected_from = None if index == 0 else ordered[index - 1].stage_id
            if gate.from_stage != expected_from:
                raise EvaluationLadderError("ladder gates must connect consecutive stages")
        body = {
            "schema_version": LADDER_SCHEMA,
            "domain_pack": descriptor.ref.to_dict(),
            "steps": [step.to_dict() for step in ordered],
        }
        return cls(descriptor.ref, ordered, canonical_sha256(body))

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationLadder:
        _exact_keys(
            value,
            {"schema_version", "domain_pack", "steps", "content_sha256"},
            "evaluation ladder",
        )
        if not isinstance(value["steps"], list):
            raise EvaluationLadderError("evaluation ladder steps must be an array")
        return cls(
            DomainPackRef.from_dict(value["domain_pack"]),
            tuple(EvaluationStep.from_dict(item) for item in value["steps"]),
            str(value["content_sha256"]),
            str(value["schema_version"]),
        )


def _result_body(
    ladder: EvaluationLadder,
    artifact: ArtifactRef,
    status: OutcomeStatus,
    stage_outcomes: Sequence[StageOutcome],
    gate_outcomes: Sequence[GateOutcome],
    skipped_steps: Sequence[str],
) -> dict[str, Any]:
    stages = tuple(stage_outcomes)
    gates = tuple(gate_outcomes)
    attempted_stage_ids = {outcome.stage_id for outcome in stages}
    attempted_steps = tuple(step for step in ladder.steps if step.stage_id in attempted_stage_ids)
    phase_counts = Counter(step.phase.value for step in attempted_steps)
    stage_status_counts = Counter(outcome.status.value for outcome in stages)
    gate_status_counts = Counter(outcome.status.value for outcome in gates)
    observational_opened = any(
        step.phase is EvaluationPhase.OBSERVATIONAL for step in attempted_steps
    )
    complete = len(gates) == len(ladder.steps) and all(
        outcome.status is OutcomeStatus.PASS for outcome in (*stages, *gates)
    )
    if complete:
        terminal = {"kind": "complete", "outcome_id": gates[-1].gate_id}
    elif len(stages) == len(gates) + 1:
        terminal = {"kind": "stage", "outcome_id": stages[-1].stage_id}
    else:
        terminal = {"kind": "gate", "outcome_id": gates[-1].gate_id}
    return {
        "schema_version": RESULT_SCHEMA,
        "ladder": ladder.to_dict(),
        "artifact": artifact.to_dict(),
        "status": status.value,
        "stage_outcomes": [item.to_dict() for item in stages],
        "gate_outcomes": [item.to_dict() for item in gates],
        "skipped_steps": list(skipped_steps),
        "terminal": terminal,
        "counts": {
            "registered_steps": len(ladder.steps),
            "attempted_stages": len(stages),
            "attempted_gates": len(gates),
            "skipped_steps": len(skipped_steps),
            "attempted_by_phase": {phase: phase_counts[phase] for phase in _PHASE_ORDER},
            "stage_statuses": {
                outcome.value: stage_status_counts[outcome.value] for outcome in OutcomeStatus
            },
            "gate_statuses": {
                outcome.value: gate_status_counts[outcome.value] for outcome in OutcomeStatus
            },
        },
        "observational_phase_opened": observational_opened,
        "all_required_gates_passed": complete,
        "claims": dict(_CLAIM_BOUNDARY),
    }


def evaluate_candidate(
    pack: DomainPack,
    artifact: CandidateArtifact,
    ladder: EvaluationLadder,
) -> dict[str, Any]:
    """Run one candidate until the first non-pass, never opening observations early."""

    artifact.validate()
    descriptor = pack.descriptor
    if ladder.domain_pack != descriptor.ref or artifact.provenance.domain_pack != descriptor.ref:
        raise EvaluationLadderError("candidate, ladder, and domain pack bindings differ")
    # Recreate against the live descriptor to reject a validly sealed but semantically unrelated plan.
    if EvaluationLadder.create(descriptor, ladder.steps) != ladder:
        raise EvaluationLadderError("evaluation ladder does not match the live descriptor")
    prior: dict[str, StageOutcome] = {}
    stages: list[StageOutcome] = []
    gates: list[GateOutcome] = []
    final_status = OutcomeStatus.PASS
    for step in ladder.steps:
        if step.phase is EvaluationPhase.OBSERVATIONAL and not any(
            planned.phase is EvaluationPhase.FORMAL
            and any(
                outcome.gate_id == planned.gate_id and outcome.status is OutcomeStatus.PASS
                for outcome in gates
            )
            for planned in ladder.steps
        ):
            raise EvaluationLadderError("observational phase reached without formal admission")
        stage_definition = descriptor.stage(step.stage_id)
        prerequisites = {
            stage_id: prior[stage_id]
            for stage_id in stage_definition.prerequisites
            if stage_id in prior
        }
        stage = run_stage(pack, artifact, step.stage_id, prerequisites)
        stages.append(stage)
        if stage.status is not OutcomeStatus.PASS:
            final_status = stage.status
            break
        prior[step.stage_id] = stage
        gate_definition = descriptor.gate(step.gate_id)
        required = {
            stage_id: prior[stage_id]
            for stage_id in gate_definition.required_stages
            if stage_id in prior
        }
        gate = run_gate(pack, artifact, step.gate_id, required)
        gates.append(gate)
        if gate.status is not OutcomeStatus.PASS:
            final_status = gate.status
            break
    attempted = {outcome.stage_id for outcome in stages}
    skipped = tuple(step.stage_id for step in ladder.steps if step.stage_id not in attempted)
    body = _result_body(ladder, artifact.ref, final_status, stages, gates, skipped)
    result = {**body, "content_sha256": canonical_sha256(body)}
    validate_evaluation_result(result)
    return result


def validate_evaluation_result(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "ladder",
        "artifact",
        "status",
        "stage_outcomes",
        "gate_outcomes",
        "skipped_steps",
        "terminal",
        "counts",
        "observational_phase_opened",
        "all_required_gates_passed",
        "claims",
        "content_sha256",
    }
    _exact_keys(value, expected, "evaluation result")
    if value["schema_version"] != RESULT_SCHEMA:
        raise EvaluationLadderError("evaluation result schema changed")
    try:
        status = OutcomeStatus(value["status"])
    except (TypeError, ValueError) as error:
        raise EvaluationLadderError("evaluation result status changed") from error
    if not all(
        isinstance(value[key], list) for key in ("stage_outcomes", "gate_outcomes", "skipped_steps")
    ):
        raise EvaluationLadderError("evaluation result collection fields must be arrays")
    ladder = EvaluationLadder.from_dict(value["ladder"])
    artifact = ArtifactRef.from_dict(value["artifact"])
    stages = tuple(StageOutcome.from_dict(item) for item in value["stage_outcomes"])
    gates = tuple(GateOutcome.from_dict(item) for item in value["gate_outcomes"])
    if any(item.artifact != artifact for item in (*stages, *gates)):
        raise EvaluationLadderError("evaluation outcomes bind a different candidate")
    if tuple(item.stage_id for item in stages) != tuple(
        step.stage_id for step in ladder.steps[: len(stages)]
    ):
        raise EvaluationLadderError("attempted stages are not a ladder prefix")
    if tuple(item.gate_id for item in gates) != tuple(
        step.gate_id for step in ladder.steps[: len(gates)]
    ):
        raise EvaluationLadderError("attempted gates are not a ladder prefix")
    if len(stages) not in {len(gates), len(gates) + 1}:
        raise EvaluationLadderError("stage and gate attempt counts are inconsistent")
    stage_refs = {outcome.stage_id: outcome.ref for outcome in stages}
    for index, gate in enumerate(gates):
        step = ladder.steps[index]
        supplied = {item.outcome_id: item for item in gate.stage_outcomes}
        if step.stage_id not in supplied or any(
            stage_id not in stage_refs or stage_refs[stage_id] != reference
            for stage_id, reference in supplied.items()
        ):
            raise EvaluationLadderError("gate outcome does not bind attempted stage outcomes")
    terminal: OutcomeStatus | None = None
    for index, stage in enumerate(stages):
        if stage.status is not OutcomeStatus.PASS:
            if index != len(stages) - 1 or len(stages) != len(gates) + 1:
                raise EvaluationLadderError("non-pass stage must be terminal")
            terminal = stage.status
    for index, gate in enumerate(gates):
        if gate.status is not OutcomeStatus.PASS:
            if index != len(gates) - 1 or len(stages) != len(gates):
                raise EvaluationLadderError("non-pass gate must be terminal")
            terminal = gate.status
    if terminal is None:
        if len(gates) != len(ladder.steps):
            raise EvaluationLadderError("passing evaluation omitted a required gate")
        terminal = OutcomeStatus.PASS
    if status is not terminal:
        raise EvaluationLadderError("evaluation result terminal status changed")
    skipped = tuple(value["skipped_steps"])
    expected_skipped = tuple(step.stage_id for step in ladder.steps[len(stages) :])
    if skipped != expected_skipped:
        raise EvaluationLadderError("evaluation skipped-step ledger changed")
    expected_body = _result_body(ladder, artifact, status, stages, gates, skipped)
    actual_body = {key: child for key, child in value.items() if key != "content_sha256"}
    if actual_body != expected_body or value["content_sha256"] != canonical_sha256(expected_body):
        raise EvaluationLadderError("evaluation result boundary changed")


def validate_evaluation_replay(
    value: Mapping[str, Any],
    pack: DomainPack,
    artifact: CandidateArtifact,
) -> None:
    """Explicitly rerun a deterministic domain pack and require byte-equivalent semantics."""

    validate_evaluation_result(value)
    ladder = EvaluationLadder.from_dict(value["ladder"])
    if evaluate_candidate(pack, artifact, ladder) != value:
        raise EvaluationLadderError("evaluation is not replayable from the supplied domain pack")


__all__ = [
    "LADDER_SCHEMA",
    "RESULT_SCHEMA",
    "EvaluationLadder",
    "EvaluationLadderError",
    "EvaluationPhase",
    "EvaluationStep",
    "evaluate_candidate",
    "validate_evaluation_replay",
    "validate_evaluation_result",
]
