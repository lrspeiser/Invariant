from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler import declarative_discovery as D


def seed(proposal_id: str, value_type: D.ValueType, representation: str) -> D.Proposal:
    return D.Proposal(
        proposal_id=proposal_id,
        declaration_id="grammar.seed",
        operator=None,
        value_type=value_type,
        representation=representation,
        parent_ids=("declaration.root",),
    )


def behavior(*, complexity: int = 2, symmetry: str = "cyclic") -> D.BehaviorDescriptor:
    return D.BehaviorDescriptor((0, 1, -1), symmetry, complexity, "polynomial", ("bounded",))


def verified(proposal: D.Proposal, *, quality: Fraction, descriptor=None) -> D.VerificationRecord:
    return D.VerificationRecord(
        proposal.proposal_id,
        "verifier.exact",
        D.VerificationStatus.VERIFIED,
        quality,
        descriptor or behavior(),
    )


def test_default_config_closes_every_declared_lane() -> None:
    config = D.load_platform_config("configs/declarative_discovery_platform.json")
    assert config.operators == tuple(D.CreativityOperator)
    assert config.dataset_stages == tuple(D.DatasetStage)
    assert config.capability_levels == tuple(D.CapabilityLevel)
    assert config.maximum_proposals == 4096


def test_declarations_are_typed_and_duplicate_symbols_fail_closed() -> None:
    declaration = D.SearchDeclaration(
        "invariant.energy",
        D.DeclarationKind.INVARIANT,
        (D.ValueType.EXPRESSION,),
        D.ValueType.EQUATION,
        symbols=(D.TypedSymbol("x", D.ValueType.SCALAR, (1, 0, -1)),),
        laws=("D_t(E)=0",),
    )
    assert declaration.to_dict()["symbols"][0]["dimension"] == [1, 0, -1]
    with pytest.raises(D.DiscoveryProtocolError):
        D.SearchDeclaration(
            "bad",
            D.DeclarationKind.GRAMMAR,
            (),
            D.ValueType.EXPRESSION,
            symbols=(
                D.TypedSymbol("x", D.ValueType.SCALAR),
                D.TypedSymbol("x", D.ValueType.SCALAR),
            ),
        )


def test_proposal_schema_cannot_smuggle_a_verdict_score_or_proof() -> None:
    proposal = seed("seed.expression", D.ValueType.EXPRESSION, "x+y")
    wire = proposal.to_dict()
    assert set(wire).isdisjoint({"verdict", "score", "proof", "behavior"})
    assert D.Proposal.from_dict(wire) == proposal
    for forbidden in ("verdict", "score", "proof", "behavior"):
        tampered = dict(wire, **{forbidden: "self-approved"})
        with pytest.raises(D.DiscoveryProtocolError):
            D.Proposal.from_dict(tampered)


def test_all_ten_creativity_families_are_executable_type_transitions() -> None:
    expression = seed("seed.expression", D.ValueType.EXPRESSION, "x^2+y^2")
    sequence = seed("seed.sequence", D.ValueType.SEQUENCE, "a")
    portfolio = D.generate_operator_portfolio((expression, sequence))
    assert {item.operator for item in portfolio} == set(D.CreativityOperator) - {
        D.CreativityOperator.COUNTEREXAMPLE_REPAIR
    }
    blocker = D.TypedBlocker(
        "blocker.row-17",
        D.BlockerKind.COUNTEREXAMPLE,
        D.ValueType.EXPRESSION,
        Fraction(3, 17),
        "x=2,y=5",
        D.CreativityOperator.COUNTEREXAMPLE_REPAIR,
    )
    repair_spec = next(
        item
        for item in D.DEFAULT_OPERATORS
        if item.operator is D.CreativityOperator.COUNTEREXAMPLE_REPAIR
    )
    repaired = D.apply_operator(expression, repair_spec, nonce="repair-1", blocker=blocker)
    assert repaired.operator is D.CreativityOperator.COUNTEREXAMPLE_REPAIR
    assert blocker.witness in repaired.representation
    assert len({item.proposal_id for item in (*portfolio, repaired)}) == 10


def test_creativity_operators_refuse_wrong_types_and_untyped_repair() -> None:
    sequence = seed("seed.sequence", D.ValueType.SEQUENCE, "a")
    dimensional = D.DEFAULT_OPERATORS[0]
    with pytest.raises(D.DiscoveryProtocolError):
        D.apply_operator(sequence, dimensional, nonce="wrong-type")
    repair = D.DEFAULT_OPERATORS[-1]
    with pytest.raises(D.DiscoveryProtocolError):
        D.apply_operator(seed("seed.x", D.ValueType.EXPRESSION, "x"), repair, nonce="no-witness")


def test_independent_verifier_is_the_only_source_of_behavior_and_quality() -> None:
    proposal = seed("seed.expression", D.ValueType.EXPRESSION, "x+y")
    registry = D.IndependentVerifierRegistry()
    registry.register(
        D.ValueType.EXPRESSION,
        "verifier.exact",
        lambda candidate: verified(candidate, quality=Fraction(7, 8)),
    )
    record = registry.verify(proposal)
    assert record.quality == Fraction(7, 8)
    assert record.behavior is not None
    with pytest.raises(D.DiscoveryProtocolError):
        registry.register(D.ValueType.EXPRESSION, "verifier.second", lambda candidate: record)


def test_behavioral_map_elites_archives_behavior_not_source_spelling() -> None:
    archive = D.BehavioralMapElites()
    first = seed("proposal-b", D.ValueType.EXPRESSION, "x+x")
    stronger = seed("proposal-a", D.ValueType.EXPRESSION, "2*x")
    distinct = seed("proposal-c", D.ValueType.EXPRESSION, "sin(x)")
    assert archive.insert(first, verified(first, quality=Fraction(1, 2)))
    assert archive.insert(stronger, verified(stronger, quality=Fraction(3, 4)))
    assert archive.occupied_niches == 1
    assert archive.elites() == (stronger,)
    assert archive.insert(
        distinct,
        verified(distinct, quality=Fraction(2, 3), descriptor=behavior(symmetry="odd")),
    )
    assert archive.occupied_niches == 2


def test_rejection_requires_a_typed_blocker_with_exact_distance() -> None:
    proposal = seed("candidate.bad", D.ValueType.EXPRESSION, "1/x")
    blocker = D.TypedBlocker(
        "blocker.zero",
        D.BlockerKind.DOMAIN_HOLE,
        D.ValueType.EXPRESSION,
        Fraction(1, 100),
        "x=0",
        D.CreativityOperator.COUNTEREXAMPLE_REPAIR,
    )
    rejected = D.VerificationRecord(
        proposal.proposal_id,
        "verifier.exact",
        D.VerificationStatus.REJECTED,
        Fraction(0),
        None,
        (blocker,),
        ("x=0",),
    )
    assert rejected.blockers[0].to_dict()["distance"] == "1/100"
    with pytest.raises(D.DiscoveryProtocolError):
        D.VerificationRecord(
            proposal.proposal_id,
            "verifier.exact",
            D.VerificationStatus.REJECTED,
            Fraction(0),
            None,
        )


def test_negative_results_require_a_valid_reachability_witness() -> None:
    certificate = D.find_type_reachability(
        D.ValueType.SEQUENCE,
        D.ValueType.EQUATION,
        "proposal-known-recurrence",
    )
    assert certificate.operator_path == (D.CreativityOperator.RECURRENCE_GUESSING,)
    result = D.publish_negative("target.no-survivors", 4000, certificate)
    assert result.status == "REAL_NEGATIVE"
    invalid = D.ReachabilityCertificate(
        D.ValueType.SCALAR,
        D.ValueType.EQUATION,
        (),
        "proposal-not-a-witness",
    )
    with pytest.raises(D.DiscoveryProtocolError):
        D.publish_negative("target.invalid", 1, invalid)


def test_proof_plan_search_finds_a_short_closed_plan_and_reports_failure() -> None:
    tactics = (
        D.TacticDeclaration("split.conjunction", "conjunction", ("algebra", "domain")),
        D.TacticDeclaration("ring.normalize", "algebra", ()),
        D.TacticDeclaration("positivity", "domain", ()),
    )
    plan = D.search_proof_plan("proposal.theorem", ("conjunction",), tactics, max_steps=4)
    assert plan.closed
    assert plan.tactic_ids == ("split.conjunction", "ring.normalize", "positivity")
    missing = D.search_proof_plan("proposal.theorem", ("analytic",), tactics)
    assert not missing.closed and missing.remaining_goals == ("analytic",)


def test_dataset_explanation_pipeline_is_staged_and_holdout_sealed() -> None:
    pipeline = D.DatasetExplanationPipeline("d" * 64, "h" * 64)
    previous = pipeline.dataset_sha256
    for stage in D.DATASET_STAGES:
        output = f"{stage.value}-sha"
        pipeline.record(
            stage,
            previous,
            output,
            passed=True,
            heldout_opened=stage is D.DatasetStage.HELDOUT_TEST,
        )
        previous = output
    assert pipeline.completed
    early = D.DatasetExplanationPipeline("d" * 64, "h" * 64)
    with pytest.raises(D.DiscoveryProtocolError):
        early.record(
            D.DatasetStage.SHAPE_AUDIT,
            "a",
            "b",
            passed=True,
            heldout_opened=True,
        )


def test_blind_capability_ladder_cannot_skip_or_leak() -> None:
    ladder = D.BlindCapabilityLadder()
    ladder.admit(D.CapabilityResult(D.CapabilityLevel.SOLVED_VISIBLE, "b1", "", False, (), True))
    ladder.admit(D.CapabilityResult(D.CapabilityLevel.SOLVED_ANONYMOUS, "b2", "", False, (), True))
    ladder.admit(
        D.CapabilityResult(
            D.CapabilityLevel.SYNTHETIC_TARGET_SEALED,
            "b3",
            "c" * 64,
            True,
            (),
            True,
        )
    )
    assert ladder.highest_passed == 3
    with pytest.raises(D.DiscoveryProtocolError):
        ladder.admit(
            D.CapabilityResult(
                D.CapabilityLevel.HISTORICAL_TARGET_SEALED,
                "b4",
                "c" * 64,
                True,
                ("target-name",),
                True,
            )
        )


def test_discovery_chain_and_creative_yield_are_replayable_without_novelty_claims() -> None:
    chain = D.DiscoveryChain()
    chain.add(D.DiscoveryChainLink("d", D.ChainStage.DECLARATION, "1" * 64, ()))
    chain.add(D.DiscoveryChainLink("p", D.ChainStage.PROPOSAL, "2" * 64, ("d",)))
    chain.add(D.DiscoveryChainLink("v", D.ChainStage.VERIFICATION, "3" * 64, ("p",)))
    assert len(chain.content_sha256()) == 64
    with pytest.raises(D.DiscoveryProtocolError):
        chain.add(D.DiscoveryChainLink("bad", D.ChainStage.PROPOSAL, "4" * 64, ("v",)))

    proposal = seed("proposal.metric", D.ValueType.EXPRESSION, "x")
    record = verified(proposal, quality=Fraction(1))
    archive = D.BehavioralMapElites()
    archive.insert(proposal, record)
    metrics = D.measure_creative_yield(
        (proposal,),
        (record,),
        archive,
        (D.ProofPlan(proposal.proposal_id, ("exact",), True, ()),),
        repairs_attempted=1,
        repairs_verified=1,
        blind_levels_passed=3,
        dataset_explanations_completed=1,
    ).to_dict()
    assert metrics["rates"]["behavioral_yield_per_proposal"] == "1/1"
    assert metrics["claims"]["novelty_established"] is False
    assert metrics["claims"]["truth_established_by_yield_metric"] is False
