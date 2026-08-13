from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.candidate_evaluation_ladder import (
    EvaluationLadder,
    EvaluationLadderError,
    EvaluationPhase,
    EvaluationStep,
    evaluate_candidate,
    validate_evaluation_replay,
    validate_evaluation_result,
)
from sigma_theory_compiler.math_expression_ir import Equation, symbol
from sigma_theory_compiler.math_pack import (
    MathDomainPack,
    math_candidate_representation,
    math_pack_descriptor,
)
from sigma_theory_compiler.math_types import IntegerType
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    CheckResult,
    DomainPackDescriptor,
    GateDefinition,
    GateOutcome,
    OutcomeStatus,
    ProvenanceRecord,
    SourceBinding,
    StageDefinition,
    StageOutcome,
    canonical_sha256,
)

KINDS = (ArtifactKind.FORMULA,)


def descriptor() -> DomainPackDescriptor:
    return DomainPackDescriptor(
        "ladder.fixture",
        "1.0",
        KINDS,
        (
            StageDefinition("cheap_screen", 0, KINDS),
            StageDefinition("symbolic_check", 1, KINDS, ("cheap_screen",)),
            StageDefinition("formal_proof", 2, KINDS, ("symbolic_check",)),
            StageDefinition("direct_observation", 3, KINDS, ("formal_proof",)),
        ),
        (
            GateDefinition("admit_cheap", None, "cheap_screen", KINDS, ("cheap_screen",)),
            GateDefinition(
                "admit_formal",
                "symbolic_check",
                "formal_proof",
                KINDS,
                ("formal_proof", "symbolic_check"),
            ),
            GateDefinition(
                "admit_observation",
                "formal_proof",
                "direct_observation",
                KINDS,
                ("direct_observation", "formal_proof"),
            ),
            GateDefinition(
                "admit_symbolic",
                "cheap_screen",
                "symbolic_check",
                KINDS,
                ("cheap_screen", "symbolic_check"),
            ),
        ),
    )


def ladder() -> EvaluationLadder:
    return EvaluationLadder.create(
        descriptor(),
        (
            EvaluationStep("cheap_screen", "admit_cheap", EvaluationPhase.CHEAP),
            EvaluationStep("symbolic_check", "admit_symbolic", EvaluationPhase.SYMBOLIC),
            EvaluationStep("formal_proof", "admit_formal", EvaluationPhase.FORMAL),
            EvaluationStep(
                "direct_observation", "admit_observation", EvaluationPhase.OBSERVATIONAL
            ),
        ),
    )


def candidate() -> CandidateArtifact:
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        "A bounded formula candidate.",
        {"operator": "add", "arguments": ["x", "1"]},
        ProvenanceRecord.create(descriptor().ref, {"fixture": 1}),
    )


class RecordingPack:
    def __init__(self, outcomes: dict[str, OutcomeStatus] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[str] = []

    @property
    def descriptor(self) -> DomainPackDescriptor:
        return descriptor()

    def _status(self, key: str) -> OutcomeStatus:
        return self.outcomes.get(key, OutcomeStatus.PASS)

    def evaluate_stage(self, artifact, stage, prior_outcomes):
        del prior_outcomes
        key = f"stage.{stage.stage_id}"
        self.calls.append(key)
        if self.outcomes.get(key) == "raise":
            raise RuntimeError("fixture explosion")
        status = self._status(key)
        return StageOutcome.create(
            stage.stage_id,
            artifact.ref,
            status,
            (CheckResult.create("stage_check", status is OutcomeStatus.PASS, {"key": key}),),
            reason_codes=() if status is OutcomeStatus.PASS else (f"{status.value}_stage",),
        )

    def evaluate_gate(self, artifact, gate, stage_outcomes):
        key = f"gate.{gate.gate_id}"
        self.calls.append(key)
        status = self._status(key)
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            status,
            tuple(stage_outcomes[item].ref for item in sorted(stage_outcomes)),
            (CheckResult.create("gate_check", status is OutcomeStatus.PASS, {"key": key}),),
            reason_codes=() if status is OutcomeStatus.PASS else (f"{status.value}_gate",),
        )


def test_all_pass_runs_every_phase_but_grants_no_promotion() -> None:
    pack = RecordingPack()
    result = evaluate_candidate(pack, candidate(), ladder())
    validate_evaluation_result(result)
    assert result["status"] == "pass"
    assert result["all_required_gates_passed"] is True
    assert result["observational_phase_opened"] is True
    assert result["counts"] == {
        "registered_steps": 4,
        "attempted_stages": 4,
        "attempted_gates": 4,
        "skipped_steps": 0,
        "attempted_by_phase": {"cheap": 1, "symbolic": 1, "formal": 1, "observational": 1},
        "stage_statuses": {"pass": 4, "block": 0, "reject": 0, "error": 0},
        "gate_statuses": {"pass": 4, "block": 0, "reject": 0, "error": 0},
    }
    assert result["terminal"] == {"kind": "complete", "outcome_id": "admit_observation"}
    assert result["claims"]["promotion_authorized_by_this_runner"] is False
    assert pack.calls[-2:] == ["stage.direct_observation", "gate.admit_observation"]


@pytest.mark.parametrize("status", [OutcomeStatus.BLOCK, OutcomeStatus.REJECT, OutcomeStatus.ERROR])
def test_formal_stage_failure_seals_observations(status: OutcomeStatus) -> None:
    pack = RecordingPack({"stage.formal_proof": status})
    result = evaluate_candidate(pack, candidate(), ladder())
    assert result["status"] == status.value
    assert result["observational_phase_opened"] is False
    assert result["skipped_steps"] == ["direct_observation"]
    assert all("observation" not in call for call in pack.calls)
    assert result["all_required_gates_passed"] is False
    assert result["terminal"] == {"kind": "stage", "outcome_id": "formal_proof"}


@pytest.mark.parametrize("status", [OutcomeStatus.BLOCK, OutcomeStatus.REJECT, OutcomeStatus.ERROR])
def test_formal_gate_failure_seals_observations(status: OutcomeStatus) -> None:
    pack = RecordingPack({"gate.admit_formal": status})
    result = evaluate_candidate(pack, candidate(), ladder())
    assert result["status"] == status.value
    assert result["observational_phase_opened"] is False
    assert all("observation" not in call for call in pack.calls)
    assert result["terminal"] == {"kind": "gate", "outcome_id": "admit_formal"}


def test_domain_pack_exception_becomes_terminal_error_without_opening_data() -> None:
    pack = RecordingPack({"stage.symbolic_check": "raise"})  # type: ignore[dict-item]
    result = evaluate_candidate(pack, candidate(), ladder())
    assert result["status"] == "error"
    assert result["stage_outcomes"][-1]["reason_codes"] == ["domain_pack_error"]
    assert result["observational_phase_opened"] is False


def test_observational_block_is_honest_about_data_phase_opening() -> None:
    pack = RecordingPack({"stage.direct_observation": OutcomeStatus.BLOCK})
    result = evaluate_candidate(pack, candidate(), ladder())
    assert result["status"] == "block"
    assert result["observational_phase_opened"] is True
    assert result["counts"]["attempted_gates"] == 3


def test_exact_replay_and_round_trip() -> None:
    result = evaluate_candidate(RecordingPack(), candidate(), ladder())
    restored = EvaluationLadder.from_dict(result["ladder"])
    assert restored == ladder()
    validate_evaluation_replay(result, RecordingPack(), candidate())
    with pytest.raises(EvaluationLadderError, match="not replayable"):
        validate_evaluation_replay(
            result,
            RecordingPack({"gate.admit_formal": OutcomeStatus.BLOCK}),
            candidate(),
        )


@pytest.mark.parametrize(
    "steps",
    [
        (
            EvaluationStep("cheap_screen", "admit_cheap", EvaluationPhase.CHEAP),
            EvaluationStep("symbolic_check", "admit_symbolic", EvaluationPhase.FORMAL),
            EvaluationStep("formal_proof", "admit_formal", EvaluationPhase.SYMBOLIC),
            EvaluationStep(
                "direct_observation", "admit_observation", EvaluationPhase.OBSERVATIONAL
            ),
        ),
        (
            EvaluationStep("cheap_screen", "admit_cheap", EvaluationPhase.CHEAP),
            EvaluationStep("symbolic_check", "admit_symbolic", EvaluationPhase.SYMBOLIC),
            EvaluationStep("formal_proof", "admit_formal", EvaluationPhase.SYMBOLIC),
            EvaluationStep(
                "direct_observation", "admit_observation", EvaluationPhase.OBSERVATIONAL
            ),
        ),
    ],
)
def test_backward_or_formal_free_observation_plan_rejects(steps) -> None:
    with pytest.raises(EvaluationLadderError):
        EvaluationLadder.create(descriptor(), steps)


def test_wrong_gate_or_incomplete_stage_plan_rejects() -> None:
    with pytest.raises(EvaluationLadderError, match="every domain stage"):
        EvaluationLadder.create(descriptor(), ladder().steps[:-1])
    bad = list(ladder().steps)
    bad[2] = EvaluationStep("formal_proof", "admit_observation", EvaluationPhase.FORMAL)
    with pytest.raises(EvaluationLadderError, match="does not admit"):
        EvaluationLadder.create(descriptor(), bad)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("observational_phase_opened",), True),
        (("all_required_gates_passed",), True),
        (("claims", "promotion_authorized_by_this_runner"), True),
        (("counts", "attempted_gates"), 4),
        (("skipped_steps",), []),
    ],
)
def test_resealed_semantic_tamper_fails_closed(path, replacement) -> None:
    value = evaluate_candidate(
        RecordingPack({"gate.admit_formal": OutcomeStatus.BLOCK}), candidate(), ladder()
    )
    tampered = copy.deepcopy(value)
    target = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    body = {key: child for key, child in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(EvaluationLadderError):
        validate_evaluation_result(tampered)


def test_unknown_result_field_and_candidate_binding_reject() -> None:
    value = evaluate_candidate(RecordingPack(), candidate(), ladder())
    value["unknown"] = True
    with pytest.raises(EvaluationLadderError, match="keys changed"):
        validate_evaluation_result(value)
    other = CandidateArtifact.create(
        ArtifactKind.FORMULA,
        "Another formula.",
        {"operator": "x"},
        ProvenanceRecord.create(descriptor().ref, {"fixture": 2}),
    )
    with pytest.raises(EvaluationLadderError, match="not replayable"):
        validate_evaluation_replay(
            evaluate_candidate(RecordingPack(), candidate(), ladder()),
            RecordingPack(),
            other,
        )


def _math_candidate(formula: Equation) -> CandidateArtifact:
    math_descriptor = math_pack_descriptor()
    return CandidateArtifact.create(
        ArtifactKind.IDENTITY,
        "The exact expressions agree on the declared integer domain.",
        math_candidate_representation(
            formula,
            {"n": IntegerType(0, 100)},
            exact_assignments=({"n": 0}, {"n": 1}, {"n": 10}),
            prior_art_receipt_sha256="a" * 64,
        ),
        ProvenanceRecord.create(
            math_descriptor.ref,
            {"integration": "evaluation_ladder"},
            sources=(SourceBinding("prior_art_receipt", "evidence/prior-art.json", "a" * 64),),
        ),
    )


def _math_ladder() -> EvaluationLadder:
    phases = {
        "typed": EvaluationPhase.CHEAP,
        "canonicalized": EvaluationPhase.CHEAP,
        "counterexample_screened": EvaluationPhase.SYMBOLIC,
        "exactly_verified": EvaluationPhase.FORMAL,
        "prior_art_checked": EvaluationPhase.FORMAL,
    }
    return EvaluationLadder.create(
        math_pack_descriptor(),
        tuple(
            EvaluationStep(stage.stage_id, f"admit_{stage.stage_id}", phases[stage.stage_id])
            for stage in math_pack_descriptor().stages
        ),
    )


def test_real_math_pack_identity_traverses_cheap_symbolic_and_formal_phases() -> None:
    n = symbol("n")
    result = evaluate_candidate(
        MathDomainPack(), _math_candidate(Equation(n * (n + 1) / 2, (n**2 + n) / 2)), _math_ladder()
    )
    assert result["status"] == "pass"
    assert result["observational_phase_opened"] is False
    assert result["counts"]["attempted_by_phase"] == {
        "cheap": 2,
        "symbolic": 1,
        "formal": 2,
        "observational": 0,
    }
    assert result["terminal"] == {
        "kind": "complete",
        "outcome_id": "admit_prior_art_checked",
    }


def test_real_math_pack_counterexample_rejects_before_formal_phase() -> None:
    n = symbol("n")
    result = evaluate_candidate(
        MathDomainPack(), _math_candidate(Equation(n * (n + 1) / 2, n**2)), _math_ladder()
    )
    assert result["status"] == "reject"
    assert result["terminal"] == {
        "kind": "stage",
        "outcome_id": "counterexample_screened",
    }
    assert result["skipped_steps"] == ["exactly_verified", "prior_art_checked"]
    assert result["counts"]["attempted_by_phase"] == {
        "cheap": 2,
        "symbolic": 1,
        "formal": 0,
        "observational": 0,
    }
