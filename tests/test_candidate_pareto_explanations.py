from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from sigma_theory_compiler.candidate_pareto_explanations import (
    MetricReceipt,
    ParetoExplanationError,
    ParetoLimits,
    build_pareto_explanations,
    validate_pareto_replay,
    validate_pareto_result,
)
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    CheckResult,
    DomainPackRef,
    GateOutcome,
    OutcomeStatus,
    ProvenanceRecord,
    SourceBinding,
    canonical_sha256,
)

GATES = ("formal", "observational")
DIRECTIONS = {"fit": "maximize", "size": "minimize"}


def _candidate(index: int) -> CandidateArtifact:
    pack = DomainPackRef("domain.pareto-test", "1.0", "a" * 64)
    provenance = ProvenanceRecord.create(pack, {"index": index})
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        f"Candidate {index}.",
        {"index": index},
        provenance,
    )


def _outcome(
    candidate: CandidateArtifact,
    gate_id: str,
    status: OutcomeStatus = OutcomeStatus.PASS,
) -> GateOutcome:
    passed = status is OutcomeStatus.PASS
    check = CheckResult.create(f"{gate_id}.check", passed, {"candidate": candidate.artifact_id})
    evidence = SourceBinding(f"{gate_id}.evidence", f"evidence/{gate_id}.json", "b" * 64)
    return GateOutcome.create(
        gate_id,
        candidate.ref,
        status,
        (),
        (check,),
        evidence=(evidence,),
        reason_codes=() if passed else (f"{gate_id}.{status.value}",),
    )


def _receipts(
    candidate: CandidateArtifact, size: int | Fraction, fit: int | Fraction
) -> tuple[MetricReceipt, MetricReceipt]:
    return (
        MetricReceipt.create(candidate.ref, "fit", "maximize", fit, canonical_sha256({"fit": fit})),
        MetricReceipt.create(
            candidate.ref, "size", "minimize", size, canonical_sha256({"size": size})
        ),
    )


def _fixture() -> tuple[
    tuple[CandidateArtifact, ...], tuple[GateOutcome, ...], tuple[MetricReceipt, ...]
]:
    candidates = tuple(_candidate(index) for index in range(4))
    outcomes = []
    for candidate in candidates[:3]:
        outcomes.extend(_outcome(candidate, gate_id) for gate_id in GATES)
    outcomes.extend(
        (
            _outcome(candidates[3], "formal", OutcomeStatus.BLOCK),
            _outcome(candidates[3], "observational"),
        )
    )
    receipts = (
        *_receipts(candidates[0], 1, 5),
        *_receipts(candidates[1], 2, 7),
        *_receipts(candidates[2], 3, 4),
        *_receipts(candidates[3], 0, 100),
    )
    return candidates, tuple(outcomes), receipts


def test_hard_gates_precede_exact_pareto_and_blocked_simplicity_never_ranks() -> None:
    candidates, outcomes, receipts = _fixture()
    result = build_pareto_explanations(
        candidates,
        outcomes,
        receipts,
        required_gate_ids=GATES,
        metric_directions=DIRECTIONS,
    )
    validate_pareto_result(result)
    fronts = [[item["artifact_id"] for item in front] for front in result["pareto_fronts"]]
    assert set(fronts[0]) == {candidates[0].artifact_id, candidates[1].artifact_id}
    assert fronts[1] == [candidates[2].artifact_id]
    blocked = next(
        row for row in result["explanations"] if row["candidate"] == candidates[3].ref.to_dict()
    )
    assert blocked["soft_metric_eligible"] is False
    assert blocked["pareto_front"] is None
    assert blocked["promotion_authorized"] is False
    assert result["display_order"][-1] == candidates[3].ref.to_dict()
    assert result["counts"] == {
        "candidates": 4,
        "hard_gate_eligible": 3,
        "hard_gate_ineligible": 1,
        "required_gates": 2,
        "metrics": 2,
        "pareto_fronts": 2,
        "work_units_consumed": 6,
    }
    assert result["claims"] == {
        "truth_established": False,
        "equivalence_established": False,
        "novelty_established": False,
        "absence_established": False,
        "promotion_authorized": False,
    }


def test_explanations_bind_gate_checks_sources_metric_receipts_and_dominators() -> None:
    candidates, outcomes, receipts = _fixture()
    result = build_pareto_explanations(
        candidates,
        outcomes,
        receipts,
        required_gate_ids=GATES,
        metric_directions=DIRECTIONS,
    )
    dominated = next(
        row for row in result["explanations"] if row["candidate"] == candidates[2].ref.to_dict()
    )
    assert {item["artifact_id"] for item in dominated["dominated_by"]} == {
        candidates[0].artifact_id,
        candidates[1].artifact_id,
    }
    assert all(row["outcome_sha256"] for row in dominated["hard_gate_outcomes"])
    assert all(row["check_details_sha256s"] for row in dominated["hard_gate_outcomes"])
    assert all(
        row["source_evidence_sha256s"] == ["b" * 64] for row in dominated["hard_gate_outcomes"]
    )
    assert all(row["receipt_sha256"] for row in dominated["metric_values"])
    assert all(item["metric_comparisons"] for item in dominated["dominance_evidence"])


def test_fraction_metrics_canonical_dedup_order_independence_and_replay() -> None:
    candidates, outcomes, receipts = _fixture()
    limits = ParetoLimits(8, 4, 4, 100)
    result = build_pareto_explanations(
        (*reversed(candidates), candidates[0]),
        tuple(reversed(outcomes)),
        tuple(reversed(receipts)),
        required_gate_ids=tuple(reversed(GATES)),
        metric_directions={"size": "minimize", "fit": "maximize"},
        limits=limits,
    )
    canonical = build_pareto_explanations(
        candidates,
        outcomes,
        receipts,
        required_gate_ids=GATES,
        metric_directions=DIRECTIONS,
        limits=limits,
    )
    assert result == canonical
    validate_pareto_replay(
        result,
        candidates,
        outcomes,
        receipts,
        required_gate_ids=GATES,
        metric_directions=DIRECTIONS,
        limits=limits,
    )
    rational = MetricReceipt.create(
        candidates[0].ref,
        "ratio",
        "maximize",
        Fraction(2, 6),
        "c" * 64,
    )
    assert rational.to_dict()["value"] == {"numerator": 1, "denominator": 3}


@pytest.mark.parametrize("status", [OutcomeStatus.BLOCK, OutcomeStatus.REJECT, OutcomeStatus.ERROR])
def test_every_nonpass_status_is_excluded_from_soft_ranking(status: OutcomeStatus) -> None:
    candidates = (_candidate(10), _candidate(11))
    outcomes = (
        _outcome(candidates[0], "formal"),
        _outcome(candidates[1], "formal", status),
    )
    receipts = (*_receipts(candidates[0], 10, 1), *_receipts(candidates[1], 0, 100))
    result = build_pareto_explanations(
        candidates,
        outcomes,
        receipts,
        required_gate_ids=("formal",),
        metric_directions=DIRECTIONS,
    )
    excluded = next(
        row for row in result["explanations"] if row["candidate"] == candidates[1].ref.to_dict()
    )
    assert excluded["pareto_front"] is None
    assert result["pareto_fronts"] == [[candidates[0].ref.to_dict()]]


def test_missing_conflicting_or_misbound_hard_gate_evidence_fails_closed() -> None:
    candidates, outcomes, receipts = _fixture()
    with pytest.raises(ParetoExplanationError, match="coverage is incomplete"):
        build_pareto_explanations(
            candidates,
            outcomes[:-1],
            receipts,
            required_gate_ids=GATES,
            metric_directions=DIRECTIONS,
        )
    conflicting = _outcome(candidates[0], "formal", OutcomeStatus.REJECT)
    with pytest.raises(ParetoExplanationError, match="conflicting gate outcomes"):
        build_pareto_explanations(
            candidates,
            (*outcomes, conflicting),
            receipts,
            required_gate_ids=GATES,
            metric_directions=DIRECTIONS,
        )
    outsider = _candidate(99)
    with pytest.raises(ParetoExplanationError, match="not bound to an input candidate"):
        build_pareto_explanations(
            candidates,
            (*outcomes, _outcome(outsider, "formal")),
            receipts,
            required_gate_ids=GATES,
            metric_directions=DIRECTIONS,
        )


def test_inexact_unknown_and_incomplete_metrics_fail_closed() -> None:
    candidates, outcomes, receipts = _fixture()
    with pytest.raises(ParetoExplanationError, match="exact integer or Fraction"):
        MetricReceipt.create(candidates[0].ref, "fit", "maximize", 0.5, "d" * 64)
    with pytest.raises(ParetoExplanationError, match="maximize or minimize"):
        MetricReceipt.create(candidates[0].ref, "fit", "smallest", 1, "d" * 64)
    with pytest.raises(ParetoExplanationError, match="coverage is incomplete"):
        build_pareto_explanations(
            candidates,
            outcomes,
            receipts[:-1],
            required_gate_ids=GATES,
            metric_directions=DIRECTIONS,
        )
    with pytest.raises(ParetoExplanationError, match="direction or registration changed"):
        build_pareto_explanations(
            candidates,
            outcomes,
            receipts,
            required_gate_ids=GATES,
            metric_directions={"fit": "minimize", "size": "minimize"},
        )


def test_candidate_metric_and_pairwise_work_bounds_fail_before_partial_ranking() -> None:
    candidates, outcomes, receipts = _fixture()
    with pytest.raises(ParetoExplanationError, match="candidate inputs exceed"):
        build_pareto_explanations(
            candidates,
            outcomes,
            receipts,
            required_gate_ids=GATES,
            metric_directions=DIRECTIONS,
            limits=ParetoLimits(3, 4, 4, 100),
        )
    with pytest.raises(ParetoExplanationError, match="metrics exceed"):
        build_pareto_explanations(
            candidates,
            outcomes,
            receipts,
            required_gate_ids=GATES,
            metric_directions=DIRECTIONS,
            limits=ParetoLimits(8, 4, 1, 100),
        )
    with pytest.raises(ParetoExplanationError, match="comparisons exceed"):
        build_pareto_explanations(
            candidates,
            outcomes,
            receipts,
            required_gate_ids=GATES,
            metric_directions=DIRECTIONS,
            limits=ParetoLimits(8, 4, 4, 5),
        )


def test_canonical_result_explanation_receipt_and_replay_tampering_is_rejected() -> None:
    candidates, outcomes, receipts = _fixture()
    result = build_pareto_explanations(
        candidates,
        outcomes,
        receipts,
        required_gate_ids=GATES,
        metric_directions=DIRECTIONS,
    )
    tampered = copy.deepcopy(result)
    tampered["explanations"][0]["promotion_authorized"] = True
    body = {
        key: value
        for key, value in tampered["explanations"][0].items()
        if key != "explanation_sha256"
    }
    tampered["explanations"][0]["explanation_sha256"] = canonical_sha256(body)
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ParetoExplanationError, match="relationships or explanations changed"):
        validate_pareto_result(tampered)

    receipt_data = receipts[0].to_dict()
    receipt_data["value"] = {"numerator": 9, "denominator": 1}
    with pytest.raises(ParetoExplanationError, match="canonical identity changed"):
        MetricReceipt.from_dict(receipt_data)

    changed = list(receipts)
    changed[0] = MetricReceipt.create(
        candidates[0].ref,
        "fit",
        "maximize",
        9,
        receipts[0].evidence_sha256,
    )
    with pytest.raises(ParetoExplanationError, match="not replayable"):
        validate_pareto_replay(
            result,
            candidates,
            outcomes,
            changed,
            required_gate_ids=GATES,
            metric_directions=DIRECTIONS,
        )
