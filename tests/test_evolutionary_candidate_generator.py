from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

import pytest

from sigma_theory_compiler.evolutionary_candidate_generator import (
    EvaluationOutcome,
    EvolutionBudget,
    EvolutionEvent,
    EvolutionRun,
    LineageRecord,
    SeedStream,
    evolve_candidates,
)
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    DomainPackDescriptor,
    GateDefinition,
    OutcomeStatus,
    ProvenanceRecord,
    SchemaViolation,
    StageDefinition,
    canonical_sha256,
)

KINDS = (ArtifactKind.FORMULA,)
DESCRIPTOR = DomainPackDescriptor(
    "test.evolution",
    "1.0.0",
    KINDS,
    (StageDefinition("typed", 0, KINDS),),
    (GateDefinition("accept_typed", None, "typed", KINDS, ("typed",)),),
)


def _candidate(
    genes: tuple[int, int],
    *,
    parents: tuple[CandidateArtifact, ...] = (),
    domain=DESCRIPTOR.ref,
) -> CandidateArtifact:
    provenance = ProvenanceRecord.create(
        domain,
        {"generator": "bounded_integer_pair", "version": 1},
        inputs=tuple(parent.ref for parent in parents),
    )
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        "A bounded anonymous integer-pair search candidate.",
        {"genes": list(genes)},
        provenance,
        assumptions=("syntactic candidate only",),
        claims=("heuristic_candidate",),
    )


def _genes(artifact: CandidateArtifact) -> tuple[int, int]:
    return tuple(artifact.representation["genes"])


def _mutate(parent: CandidateArtifact, stream: SeedStream) -> CandidateArtifact:
    genes = list(_genes(parent))
    index = stream.draw(2)
    genes[index] = (genes[index] + 1 + stream.draw(3)) % 11
    return _candidate(tuple(genes), parents=(parent,))


def _crossover(
    left: CandidateArtifact, right: CandidateArtifact, stream: SeedStream
) -> CandidateArtifact:
    left_genes = _genes(left)
    right_genes = _genes(right)
    if stream.draw(2) == 0:
        genes = (left_genes[0], right_genes[1])
    else:
        genes = (right_genes[0], left_genes[1])
    return _candidate(genes, parents=(left, right))


def _evaluate(artifact: CandidateArtifact) -> EvaluationOutcome:
    left, right = _genes(artifact)
    total = left + right
    if total == 13:
        raise RuntimeError("private evaluator detail")
    if total > 15:
        return EvaluationOutcome.create(
            artifact, OutcomeStatus.REJECT, reason_codes=("outside_declared_domain",)
        )
    if left == right:
        return EvaluationOutcome.create(
            artifact, OutcomeStatus.BLOCK, reason_codes=("degenerate_pair",)
        )
    return EvaluationOutcome.create(artifact, OutcomeStatus.PASS, score=100 - total)


def _run() -> EvolutionRun:
    return evolve_candidates(
        (_candidate((1, 4)), _candidate((2, 7))),
        seed="deterministic-evolution-seed-001",
        budget=EvolutionBudget(
            population_size=4,
            generations=3,
            offspring_per_generation=4,
            max_evaluations=10,
        ),
        mutate=_mutate,
        crossover=_crossover,
        evaluate=_evaluate,
    )


def _reseal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return {**body, "content_sha256": canonical_sha256(body)}


def test_deterministic_seeded_evolution_replays_byte_exactly() -> None:
    first = _run()
    second = _run()
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.content_sha256 == canonical_sha256(
        {key: item for key, item in first.to_dict().items() if key != "content_sha256"}
    )
    assert EvolutionRun.from_dict(first.to_dict()) == first
    assert first.seed_sha256 == hashlib.sha256(b"deterministic-evolution-seed-001").hexdigest()


def test_mutation_and_crossover_have_exact_candidate_provenance_lineage() -> None:
    run = _run()
    operations = {record.operation for record in run.lineage}
    assert operations == {"seed", "mutation", "crossover"}
    artifacts = {item.content_sha256: item for item in run.artifacts}
    for record in run.lineage:
        child = artifacts[record.child.content_sha256]
        expected = () if record.operation == "seed" else record.parents
        assert child.provenance.inputs == expected
        assert record.child == child.ref
        assert LineageRecord.from_dict(record.to_dict()) == record


def test_population_is_canonical_content_hash_deduplicated_and_ranked() -> None:
    duplicate = _candidate((1, 4))
    run = evolve_candidates(
        (duplicate, duplicate),
        seed="dedup-seed",
        budget=EvolutionBudget(2, 0, 0, 2),
        mutate=_mutate,
        crossover=_crossover,
        evaluate=_evaluate,
    )
    assert len(run.artifacts) == len(run.lineage) == len(run.evaluations) == 1
    assert run.counts["deduplicated"] == 1
    assert run.counts["evaluations"] == 1
    assert [item.content_sha256 for item in run.artifacts] == sorted(
        item.content_sha256 for item in run.artifacts
    )
    assert run.final_population == (duplicate.ref,)


def test_budgets_bound_operator_attempts_evaluations_and_population() -> None:
    budget = EvolutionBudget(3, 20, 7, 5)
    run = evolve_candidates(
        (_candidate((1, 2)), _candidate((3, 4))),
        seed="tight-budget",
        budget=budget,
        mutate=_mutate,
        crossover=_crossover,
        evaluate=_evaluate,
    )
    assert run.counts["evaluations"] == budget.max_evaluations
    assert run.counts["operator_attempts"] <= budget.max_operator_attempts
    assert len(run.final_population) <= budget.population_size
    assert len(run.events) <= len(run.artifacts) + budget.max_operator_attempts


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 1, 1),
        (1, -1, 1, 1),
        (1, 1, -1, 1),
        (1, 1, 1, 0),
        (True, 1, 1, 1),
    ],
)
def test_budget_rejects_impossible_and_boolean_counts(values: tuple[int, int, int, int]) -> None:
    with pytest.raises(SchemaViolation):
        EvolutionBudget(*values)


def test_initial_population_and_domain_boundaries_fail_closed() -> None:
    budget = EvolutionBudget(1, 0, 0, 1)
    with pytest.raises(SchemaViolation, match="nonempty"):
        evolve_candidates(
            (),
            seed="x",
            budget=budget,
            mutate=_mutate,
            crossover=_crossover,
            evaluate=_evaluate,
        )
    other = DomainPackDescriptor(
        "test.other",
        "1.0.0",
        KINDS,
        (StageDefinition("typed", 0, KINDS),),
        (GateDefinition("accept_typed", None, "typed", KINDS, ("typed",)),),
    )
    with pytest.raises(SchemaViolation, match="multiple domain packs"):
        evolve_candidates(
            (_candidate((1, 2)), _candidate((2, 3), domain=other.ref)),
            seed="x",
            budget=EvolutionBudget(2, 0, 0, 2),
            mutate=_mutate,
            crossover=_crossover,
            evaluate=_evaluate,
        )


def test_invalid_mutation_lineage_is_typed_reject_not_an_exception() -> None:
    def bad_mutation(parent: CandidateArtifact, stream: SeedStream) -> CandidateArtifact:
        del stream
        return _candidate((9, 1))  # Missing the required parent provenance input.

    run = evolve_candidates(
        (_candidate((1, 4)),),
        seed="bad-lineage",
        budget=EvolutionBudget(2, 1, 1, 2),
        mutate=bad_mutation,
        crossover=_crossover,
        evaluate=_evaluate,
    )
    rejected = [event for event in run.events if event.status is OutcomeStatus.REJECT]
    assert len(rejected) == 1
    assert rejected[0].reason_codes == ("parent_lineage_mismatch",)
    assert run.counts["event_reject"] == 1
    assert run.counts["evaluations"] == 1


def test_operator_and_evaluator_exceptions_become_error_outcomes_without_details() -> None:
    def exploding_mutation(parent: CandidateArtifact, stream: SeedStream) -> CandidateArtifact:
        del parent, stream
        raise RuntimeError("secret operator detail")

    operator_run = evolve_candidates(
        (_candidate((1, 4)),),
        seed="operator-error",
        budget=EvolutionBudget(2, 1, 1, 2),
        mutate=exploding_mutation,
        crossover=_crossover,
        evaluate=_evaluate,
    )
    error_event = operator_run.events[-1]
    assert error_event.status is OutcomeStatus.ERROR
    assert error_event.reason_codes == ("operator_exception",)
    assert "secret" not in str(operator_run.to_dict())

    evaluator_run = evolve_candidates(
        (_candidate((6, 7)),),
        seed="evaluator-error",
        budget=EvolutionBudget(1, 0, 0, 1),
        mutate=_mutate,
        crossover=_crossover,
        evaluate=_evaluate,
    )
    assert evaluator_run.evaluations[0].status is OutcomeStatus.ERROR
    assert evaluator_run.evaluations[0].reason_codes == ("evaluator_exception",)
    assert evaluator_run.counts["error"] == 1
    assert evaluator_run.final_population == ()
    assert "private" not in str(evaluator_run.to_dict())


@pytest.mark.parametrize(
    ("genes", "status", "reason"),
    [
        ((4, 4), OutcomeStatus.BLOCK, "degenerate_pair"),
        ((9, 8), OutcomeStatus.REJECT, "outside_declared_domain"),
        ((6, 7), OutcomeStatus.ERROR, "evaluator_exception"),
    ],
)
def test_block_reject_and_error_are_distinct_typed_fail_closed_outcomes(
    genes: tuple[int, int], status: OutcomeStatus, reason: str
) -> None:
    run = evolve_candidates(
        (_candidate(genes),),
        seed=f"status-{status.value}",
        budget=EvolutionBudget(1, 0, 0, 1),
        mutate=_mutate,
        crossover=_crossover,
        evaluate=_evaluate,
    )
    assert run.evaluations[0].status is status
    assert run.evaluations[0].reason_codes == (reason,)
    assert run.events[0].status is status
    assert run.final_population == ()


def test_invalid_evaluator_return_is_an_error_outcome() -> None:
    run = evolve_candidates(
        (_candidate((1, 4)),),
        seed="invalid-evaluator",
        budget=EvolutionBudget(1, 0, 0, 1),
        mutate=_mutate,
        crossover=_crossover,
        evaluate=lambda artifact: artifact,  # type: ignore[arg-type,return-value]
    )
    assert run.evaluations[0].status is OutcomeStatus.ERROR
    assert run.evaluations[0].reason_codes == ("invalid_evaluator_outcome",)


def test_selection_pass_explicitly_cannot_claim_truth_or_promotion() -> None:
    run = _run()
    encoded = run.to_dict()
    assert encoded["selection_scope"] == "bounded_heuristic_population_search_only"
    assert encoded["physics_truth_established"] is False
    assert encoded["promotion_allowed"] is False
    assert all(item.to_dict()["selection_only"] is True for item in run.evaluations)
    assert all(item.to_dict()["physics_truth_established"] is False for item in run.evaluations)
    assert all(item.to_dict()["promotion_allowed"] is False for item in run.evaluations)


def test_signed_integer_scores_are_allowed_but_boolean_scores_fail_closed() -> None:
    artifact = _candidate((1, 4))
    outcome = EvaluationOutcome.create(artifact, OutcomeStatus.PASS, score=-7)
    assert outcome.score == -7
    with pytest.raises(SchemaViolation, match="score"):
        EvaluationOutcome.create(artifact, OutcomeStatus.PASS, score=True)


def test_nested_and_resealed_run_tampering_fails_closed() -> None:
    run = _run()
    forged = deepcopy(run.to_dict())
    forged["lineage"][-1]["operation"] = "seed"
    forged = _reseal(forged)
    with pytest.raises(SchemaViolation):
        EvolutionRun.from_dict(forged)

    forged = deepcopy(run.to_dict())
    forged["artifacts"][0]["representation"]["genes"][0] += 1
    forged = _reseal(forged)
    with pytest.raises(SchemaViolation):
        EvolutionRun.from_dict(forged)

    forged = deepcopy(run.to_dict())
    forged["counts"]["pass"] += 1
    forged = _reseal(forged)
    with pytest.raises(SchemaViolation, match="counts"):
        EvolutionRun.from_dict(forged)

    forged = deepcopy(run.to_dict())
    forged["promotion_allowed"] = True
    forged = _reseal(forged)
    with pytest.raises(SchemaViolation, match="scientific"):
        EvolutionRun.from_dict(forged)


def test_outcome_lineage_and_event_individual_tampering_fails_closed() -> None:
    run = _run()
    evaluation = deepcopy(run.evaluations[0].to_dict())
    evaluation["score"] += 1
    with pytest.raises(SchemaViolation, match="canonical identity"):
        EvaluationOutcome.from_dict(evaluation)

    lineage = deepcopy(run.lineage[-1].to_dict())
    lineage["generation"] += 1
    with pytest.raises(SchemaViolation, match="canonical identity"):
        LineageRecord.from_dict(lineage)

    event = deepcopy(run.events[-1].to_dict())
    event["ordinal"] += 1
    with pytest.raises(SchemaViolation, match="canonical identity"):
        EvolutionEvent.from_dict(event)


def test_seed_stream_is_deterministic_bounded_and_rejects_bad_bounds() -> None:
    left = SeedStream("stream")
    right = SeedStream("stream")
    assert [left.draw(7) for _ in range(20)] == [right.draw(7) for _ in range(20)]
    assert left.draws == right.draws == 20
    with pytest.raises(SchemaViolation):
        left.draw(0)
