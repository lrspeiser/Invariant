from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from sigma_theory_compiler.bayesian_candidate_generator import (
    SCOPE,
    BayesianBudget,
    BayesianCandidateGenerator,
    BayesianGeneratorError,
    BayesianProposalBatch,
    BayesianState,
    EvidenceBatch,
    EvidenceLikelihood,
    ExactProbability,
    WeightedCandidate,
)
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    DomainPackRef,
    ProvenanceRecord,
    canonical_sha256,
)

PACK = DomainPackRef("math.test", "1.0", "1" * 64)
BUDGET = BayesianBudget(max_candidates=4, max_evidence_updates=2, max_proposal_draws=32)


def _candidate(index: int) -> CandidateArtifact:
    provenance = ProvenanceRecord.create(
        PACK,
        {"generator": "bayesian-test", "index": index},
    )
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        f"f_{index}(x) = {index} * x",
        {"node": "multiply", "arguments": [index, {"symbol": "x"}]},
        provenance,
        assumptions=("x is an exact scalar",),
        claims=("candidate_formula",),
    )


def _weighted() -> tuple[WeightedCandidate, ...]:
    return (
        WeightedCandidate(_candidate(1), ExactProbability(1, 3)),
        WeightedCandidate(_candidate(2), ExactProbability(2, 3)),
    )


def _evidence(
    state: BayesianState, first: ExactProbability, second: ExactProbability
) -> EvidenceBatch:
    return EvidenceBatch.create(
        "observation-1",
        (
            EvidenceLikelihood(state.candidates[0].artifact.artifact_id, first),
            EvidenceLikelihood(state.candidates[1].artifact.artifact_id, second),
        ),
    )


def test_exact_prior_normalization_dedup_and_sigma_provenance() -> None:
    first, second = _weighted()
    state = BayesianState.create((first, first, second), BUDGET)

    assert len(state.candidates) == 2
    assert state.deduplicated_input_count == 1
    assert [item.prior.fraction for item in state.candidates] == [Fraction(1, 2), Fraction(1, 2)]
    assert all(isinstance(item.artifact.provenance, ProvenanceRecord) for item in state.candidates)
    assert BayesianState.from_dict(state.to_dict()) == state
    assert state.scope == SCOPE
    assert "truth" not in state.to_dict()
    assert "promotion" not in state.to_dict()


def test_exact_posterior_update_and_append_only_lineage() -> None:
    weighted = _weighted()
    state = BayesianState.create(weighted, BUDGET)
    evidence = EvidenceBatch.create(
        "observation-1",
        (
            EvidenceLikelihood(weighted[0].artifact.artifact_id, ExactProbability(1, 2)),
            EvidenceLikelihood(weighted[1].artifact.artifact_id, ExactProbability(1, 4)),
        ),
    )

    updated = state.update(evidence)

    prior_by_id = {item.artifact.artifact_id: item.prior.fraction for item in updated.candidates}
    assert prior_by_id == {
        weighted[0].artifact.artifact_id: Fraction(1, 3),
        weighted[1].artifact.artifact_id: Fraction(2, 3),
    }
    assert {item.posterior.fraction for item in updated.candidates} == {Fraction(1, 2)}
    assert updated.parent_state_sha256 == state.content_sha256
    assert updated.content_sha256 != state.content_sha256
    assert updated.lineage_sha256 != state.lineage_sha256
    assert updated.evidence_history == (evidence,)


def test_zero_likelihood_and_incomplete_likelihood_fail_closed_without_mutation() -> None:
    state = BayesianState.create(_weighted(), BUDGET)
    zero = _evidence(state, ExactProbability(0, 1), ExactProbability(0, 1))
    incomplete = EvidenceBatch.create(
        "observation-incomplete",
        (EvidenceLikelihood(state.candidates[0].artifact.artifact_id, ExactProbability(1, 1)),),
    )

    with pytest.raises(BayesianGeneratorError, match="probability mass is zero"):
        state.update(zero)
    with pytest.raises(BayesianGeneratorError, match="exactly the candidate state"):
        state.update(incomplete)

    assert state.evidence_history == ()
    assert BayesianState.from_dict(state.to_dict()) == state


def test_evidence_replay_and_update_budget_fail_closed() -> None:
    one_update_budget = BayesianBudget(4, 1, 8)
    state = BayesianState.create(_weighted(), one_update_budget)
    evidence = _evidence(state, ExactProbability(1, 1), ExactProbability(1, 2))
    updated = state.update(evidence)

    with pytest.raises(BayesianGeneratorError, match="max_evidence_updates exhausted"):
        updated.update(evidence)

    replay_budget = BayesianBudget(4, 2, 8)
    replay_state = BayesianState.create(_weighted(), replay_budget)
    replay_updated = replay_state.update(evidence)
    with pytest.raises(BayesianGeneratorError, match="evidence_id replay"):
        replay_updated.update(evidence)


def test_seeded_proposals_are_exact_deterministic_replayable_and_deduplicated() -> None:
    state = BayesianState.create(_weighted(), BUDGET)

    first = BayesianCandidateGenerator.propose(state, seed=20260812, draws=16)
    second = BayesianCandidateGenerator.propose(state, seed=20260812, draws=16)

    assert first == second
    assert BayesianProposalBatch.from_dict(first.to_dict()) == first
    assert first.requested_draws == 16
    assert first.duplicates_removed == 16 - len(first.proposals)
    assert len(first.proposals) == len({item.content_sha256 for item in first.proposals})
    assert first.source_state_sha256 == state.content_sha256
    assert first.scope == SCOPE


def test_different_seeds_bind_distinct_proposal_lineage() -> None:
    state = BayesianState.create(_weighted(), BUDGET)
    first = BayesianCandidateGenerator.propose(state, seed=7, draws=16)
    second = BayesianCandidateGenerator.propose(state, seed=8, draws=16)

    assert first.lineage_sha256 != second.lineage_sha256
    assert first.content_sha256 != second.content_sha256


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ExactProbability(1, 2.0), "components must be integers"),
        (lambda: ExactProbability(2, 4), "lowest terms"),
        (lambda: ExactProbability(-1, 2), "must lie in"),
        (lambda: BayesianBudget(0, 1, 1), "max_candidates"),
        (lambda: BayesianBudget(1, 1, 100_001), "max_proposal_draws"),
    ],
)
def test_invalid_numeric_and_budget_inputs_fail_closed(factory: object, message: str) -> None:
    with pytest.raises(BayesianGeneratorError, match=message):
        factory()  # type: ignore[operator]


def test_candidate_and_proposal_budgets_are_enforced() -> None:
    with pytest.raises(BayesianGeneratorError, match="exceed max_candidates"):
        BayesianState.create(
            _weighted(),
            BayesianBudget(max_candidates=1, max_evidence_updates=1, max_proposal_draws=1),
        )

    state = BayesianState.create(_weighted(), BUDGET)
    with pytest.raises(BayesianGeneratorError, match="draws"):
        BayesianCandidateGenerator.propose(state, seed=0, draws=33)
    with pytest.raises(BayesianGeneratorError, match="seed"):
        BayesianCandidateGenerator.propose(state, seed=-1, draws=1)


def test_nested_state_tamper_and_unknown_keys_are_rejected() -> None:
    state = BayesianState.create(_weighted(), BUDGET)
    tampered = copy.deepcopy(state.to_dict())
    tampered["candidates"][0]["artifact"]["representation"]["arguments"][0] = 999
    with pytest.raises(BayesianGeneratorError, match="candidate artifact"):
        BayesianState.from_dict(tampered)

    resealed = copy.deepcopy(state.to_dict())
    resealed["scope"] = "posterior proves truth"
    body = {key: value for key, value in resealed.items() if key != "content_sha256"}
    resealed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(BayesianGeneratorError, match="schema or scope"):
        BayesianState.from_dict(resealed)

    unknown = copy.deepcopy(state.to_dict())
    unknown["promotion_eligible"] = True
    with pytest.raises(BayesianGeneratorError, match="keys changed"):
        BayesianState.from_dict(unknown)


def test_resealed_evidence_and_proposal_semantic_tamper_is_rejected() -> None:
    state = BayesianState.create(_weighted(), BUDGET)
    proposal = BayesianCandidateGenerator.propose(state, seed=41, draws=8)
    tampered_proposal = copy.deepcopy(proposal.to_dict())
    tampered_proposal["draw_artifact_ids"][0] = "sig-forged"
    lineage_body = {
        "schema_version": tampered_proposal["schema_version"],
        "source_state_sha256": tampered_proposal["source_state_sha256"],
        "seed": tampered_proposal["seed"],
        "draw_artifact_ids": tampered_proposal["draw_artifact_ids"],
    }
    tampered_proposal["lineage_sha256"] = canonical_sha256(lineage_body)
    body = {key: value for key, value in tampered_proposal.items() if key != "content_sha256"}
    tampered_proposal["content_sha256"] = canonical_sha256(body)
    with pytest.raises(BayesianGeneratorError, match="dedup boundary"):
        BayesianProposalBatch.from_dict(tampered_proposal)

    evidence = _evidence(state, ExactProbability(1, 2), ExactProbability(1, 4))
    tampered_evidence = copy.deepcopy(evidence.to_dict())
    tampered_evidence["likelihoods"].reverse()
    evidence_body = {
        key: value for key, value in tampered_evidence.items() if key != "content_sha256"
    }
    tampered_evidence["content_sha256"] = canonical_sha256(evidence_body)
    with pytest.raises(BayesianGeneratorError, match="unique and sorted"):
        EvidenceBatch.from_dict(tampered_evidence)
