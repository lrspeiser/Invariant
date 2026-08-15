from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from sigma_theory_compiler.candidate_discovery_pipeline import (
    CandidateDiscoveryPipelineError,
    DiscoveryPipelineLimits,
    run_candidate_discovery_pipeline,
    validate_candidate_discovery_replay,
    validate_candidate_discovery_result,
)
from sigma_theory_compiler.candidate_evaluation_ladder import (
    EvaluationLadder,
    EvaluationPhase,
    EvaluationStep,
)
from sigma_theory_compiler.candidate_pareto_explanations import MetricReceipt, ParetoLimits
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    CheckResult,
    DomainPackDescriptor,
    GateDefinition,
    GateOutcome,
    OutcomeStatus,
    ProvenanceRecord,
    StageDefinition,
    StageOutcome,
    canonical_sha256,
)

KINDS = (ArtifactKind.FORMULA,)


def descriptor() -> DomainPackDescriptor:
    return DomainPackDescriptor(
        "pipeline.fixture",
        "1.0",
        KINDS,
        (
            StageDefinition("cheap", 0, KINDS),
            StageDefinition("formal", 1, KINDS, ("cheap",)),
            StageDefinition("observation", 2, KINDS, ("formal",)),
        ),
        (
            GateDefinition("admit_cheap", None, "cheap", KINDS, ("cheap",)),
            GateDefinition("admit_formal", "cheap", "formal", KINDS, ("cheap", "formal")),
            GateDefinition(
                "admit_observation",
                "formal",
                "observation",
                KINDS,
                ("formal", "observation"),
            ),
        ),
    )


def ladder() -> EvaluationLadder:
    return EvaluationLadder.create(
        descriptor(),
        (
            EvaluationStep("cheap", "admit_cheap", EvaluationPhase.CHEAP),
            EvaluationStep("formal", "admit_formal", EvaluationPhase.FORMAL),
            EvaluationStep("observation", "admit_observation", EvaluationPhase.OBSERVATIONAL),
        ),
    )


def candidate(name: str, mode: str) -> CandidateArtifact:
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        f"candidate {name}",
        {"name": name, "mode": mode},
        ProvenanceRecord.create(descriptor().ref, {"name": name}),
    )


class FixturePack:
    @property
    def descriptor(self) -> DomainPackDescriptor:
        return descriptor()

    def evaluate_stage(self, artifact, stage, prior_outcomes):
        del prior_outcomes
        mode = artifact.representation["mode"]
        status = OutcomeStatus.PASS
        if stage.stage_id == "cheap" and mode == "reject_cheap":
            status = OutcomeStatus.REJECT
        if stage.stage_id == "formal" and mode == "block_formal":
            status = OutcomeStatus.BLOCK
        return StageOutcome.create(
            stage.stage_id,
            artifact.ref,
            status,
            (CheckResult.create("fixture_stage", status is OutcomeStatus.PASS, {"mode": mode}),),
            reason_codes=() if status is OutcomeStatus.PASS else (f"{mode}_{stage.stage_id}",),
        )

    def evaluate_gate(self, artifact, gate, stage_outcomes):
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            OutcomeStatus.PASS,
            tuple(stage_outcomes[key].ref for key in sorted(stage_outcomes)),
            (CheckResult.create("fixture_gate", True, {"gate": gate.gate_id}),),
        )


def receipt(item: CandidateArtifact, value: int | Fraction) -> MetricReceipt:
    return MetricReceipt.create(
        item.ref,
        "simplicity",
        "minimize",
        value,
        canonical_sha256({"candidate": item.artifact_id, "metric": "simplicity"}),
    )


def fixture():
    first = candidate("passing-a", "pass")
    second = candidate("passing-b", "pass")
    blocked = candidate("blocked", "block_formal")
    rejected = candidate("rejected", "reject_cheap")
    metrics = (receipt(first, 1), receipt(second, 2))
    return (first, second, blocked, rejected), metrics


def test_hard_gates_precede_metrics_and_failed_candidates_remain_unranked() -> None:
    candidates, metrics = fixture()
    result = run_candidate_discovery_pipeline(
        FixturePack(),
        candidates,
        ladder(),
        metrics,
        metric_directions={"simplicity": "minimize"},
    )
    validate_candidate_discovery_result(result)
    assert result["counts"] == {
        "candidates": 4,
        "evaluation_statuses": {"pass": 2, "block": 1, "reject": 1, "error": 0},
        "all_required_gates_passed": 2,
        "ranked_candidates": 2,
        "unranked_candidates": 2,
        "observational_phase_opened": 2,
        "metric_receipts": 2,
        "pareto_fronts": 2,
    }
    rows = {row["candidate"]["artifact_id"]: row for row in result["candidate_rows"]}
    for candidate_item in candidates[:2]:
        assert rows[candidate_item.artifact_id]["ranking_eligible"] is True
        assert rows[candidate_item.artifact_id]["pareto_front"] in {1, 2}
    for candidate_item in candidates[2:]:
        row = rows[candidate_item.artifact_id]
        assert row["ranking_eligible"] is False
        assert row["pareto_front"] is None
        assert row["explanation_sha256"] is None
        assert row["exclusion_reason"] == "did_not_pass_all_required_hard_gates"
    assert result["claims"] == {
        "soft_metric_receipts_admitted_before_all_hard_gates_passed": False,
        "failed_hard_gate_compensated_by_soft_metric": False,
        "unranked_candidate_omitted": False,
        "truth_established": False,
        "novelty_established": False,
        "promotion_authorized": False,
    }


def test_exact_metric_orders_the_two_hard_gate_survivors() -> None:
    candidates, metrics = fixture()
    result = run_candidate_discovery_pipeline(
        FixturePack(),
        candidates,
        ladder(),
        metrics,
        metric_directions={"simplicity": "minimize"},
    )
    rows = {row["candidate"]["artifact_id"]: row for row in result["candidate_rows"]}
    assert rows[candidates[0].artifact_id]["pareto_front"] == 1
    assert rows[candidates[1].artifact_id]["pareto_front"] == 2
    assert all(row["promotion_authorized"] is False for row in rows.values())


def test_no_survivor_produces_no_pareto_or_soft_receipts() -> None:
    candidates = (
        candidate("blocked-only", "block_formal"),
        candidate("rejected-only", "reject_cheap"),
    )
    result = run_candidate_discovery_pipeline(
        FixturePack(),
        candidates,
        ladder(),
        (),
        metric_directions={"simplicity": "minimize"},
    )
    assert result["pareto_result"] is None
    assert result["metric_receipts"] == []
    assert result["decision"] == "completed_with_no_candidate_admitted_to_soft_ranking"
    assert result["counts"]["ranked_candidates"] == 0


def test_metric_for_failed_candidate_is_rejected_before_ranking() -> None:
    passing = candidate("passing", "pass")
    blocked = candidate("blocked-metric", "block_formal")
    with pytest.raises(CandidateDiscoveryPipelineError, match="did not pass every hard gate"):
        run_candidate_discovery_pipeline(
            FixturePack(),
            (passing, blocked),
            ladder(),
            (receipt(passing, 1), receipt(blocked, 0)),
            metric_directions={"simplicity": "minimize"},
        )


def test_missing_survivor_metric_fails_closed() -> None:
    candidates, metrics = fixture()
    with pytest.raises(CandidateDiscoveryPipelineError, match="lack complete metric"):
        run_candidate_discovery_pipeline(
            FixturePack(),
            candidates,
            ladder(),
            metrics[:1],
            metric_directions={"simplicity": "minimize"},
        )


def test_replay_recomputes_every_stage_gate_and_pareto_front() -> None:
    candidates, metrics = fixture()
    result = run_candidate_discovery_pipeline(
        FixturePack(),
        candidates,
        ladder(),
        metrics,
        metric_directions={"simplicity": "minimize"},
    )
    validate_candidate_discovery_replay(result, FixturePack(), candidates, metrics)
    wrong_metrics = (receipt(candidates[0], 3), metrics[1])
    with pytest.raises(CandidateDiscoveryPipelineError, match="not replayable"):
        validate_candidate_discovery_replay(result, FixturePack(), candidates, wrong_metrics)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["candidate_rows"][0].__setitem__("promotion_authorized", True),
            "identity changed",
        ),
        (
            lambda value: value["claims"].__setitem__(
                "failed_hard_gate_compensated_by_soft_metric", True
            ),
            "identity changed",
        ),
        (
            lambda value: value["counts"].__setitem__("ranked_candidates", 4),
            "identity changed",
        ),
    ],
)
def test_unsealed_tampering_fails_closed(mutation, message: str) -> None:
    candidates, metrics = fixture()
    result = run_candidate_discovery_pipeline(
        FixturePack(),
        candidates,
        ladder(),
        metrics,
        metric_directions={"simplicity": "minimize"},
    )
    broken = copy.deepcopy(result)
    mutation(broken)
    with pytest.raises(CandidateDiscoveryPipelineError, match=message):
        validate_candidate_discovery_result(broken)


def test_resealed_semantic_tampering_fails_closed() -> None:
    candidates, metrics = fixture()
    result = run_candidate_discovery_pipeline(
        FixturePack(),
        candidates,
        ladder(),
        metrics,
        metric_directions={"simplicity": "minimize"},
    )
    broken = copy.deepcopy(result)
    broken["candidate_rows"][0]["pareto_front"] = 99
    unsigned = {key: value for key, value in broken.items() if key != "content_sha256"}
    broken["content_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(CandidateDiscoveryPipelineError, match="boundary changed"):
        validate_candidate_discovery_result(broken)


def test_resealed_pareto_transplant_is_rejected_against_batch_evidence() -> None:
    candidates, metrics = fixture()
    result = run_candidate_discovery_pipeline(
        FixturePack(),
        candidates,
        ladder(),
        metrics,
        metric_directions={"simplicity": "minimize"},
    )
    alternate = run_candidate_discovery_pipeline(
        FixturePack(),
        candidates,
        ladder(),
        (receipt(candidates[0], 3), metrics[1]),
        metric_directions={"simplicity": "minimize"},
    )
    broken = copy.deepcopy(result)
    broken["pareto_result"] = alternate["pareto_result"]
    unsigned = {key: value for key, value in broken.items() if key != "content_sha256"}
    broken["content_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(CandidateDiscoveryPipelineError, match="not bound"):
        validate_candidate_discovery_result(broken)


def test_duplicate_candidate_and_capacity_fail_before_evaluation() -> None:
    item = candidate("duplicate", "pass")
    with pytest.raises(CandidateDiscoveryPipelineError, match="duplicate"):
        run_candidate_discovery_pipeline(
            FixturePack(),
            (item, item),
            ladder(),
            (),
            metric_directions={"simplicity": "minimize"},
        )
    with pytest.raises(CandidateDiscoveryPipelineError, match="exceeds"):
        run_candidate_discovery_pipeline(
            FixturePack(),
            (item, candidate("second", "pass")),
            ladder(),
            (),
            metric_directions={"simplicity": "minimize"},
            limits=DiscoveryPipelineLimits(maximum_candidates=1),
            pareto_limits=ParetoLimits(),
        )
