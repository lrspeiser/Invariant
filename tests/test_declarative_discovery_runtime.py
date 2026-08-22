from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler import declarative_discovery as D
from sigma_theory_compiler import declarative_discovery_runtime as R
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def candidates() -> tuple[R.ExtensionCandidate, ...]:
    value = json.loads(
        (ROOT / "configs/declarative_extension_candidates.json").read_text(encoding="utf-8")
    )
    assert value["schema_version"] == "invariant-declarative-extension-set-2.0"
    return tuple(R.ExtensionCandidate.from_dict(item) for item in value["candidates"])


def identity(
    verifier_id: str, principal_id: str, backend: R.VerifierBackend
) -> R.VerifierIdentity:
    return R.VerifierIdentity(
        verifier_id,
        principal_id,
        backend,
        canonical_sha256({"implementation": verifier_id}),
    )


def admission_registry() -> tuple[R.ExtensionVerifierRegistry, R.ExtensionAdmissionRegistry]:
    verifiers = R.ExtensionVerifierRegistry()
    verifiers.register(
        identity("verifier.extension-schema", "principal.schema-team", R.VerifierBackend.SCHEMA),
        R.structural_extension_verifier,
    )
    verifiers.register(
        identity(
            "verifier.extension-replay",
            "principal.replay-team",
            R.VerifierBackend.EXACT_ARITHMETIC,
        ),
        R.replay_extension_verifier,
    )
    admissions = R.ExtensionAdmissionRegistry(
        R.AdmissionPolicy(
            (R.VerifierBackend.SCHEMA, R.VerifierBackend.EXACT_ARITHMETIC), 2
        )
    )
    return verifiers, admissions


def seed(
    proposal_id: str, value_type: D.ValueType, representation: str
) -> D.Proposal:
    return D.Proposal(
        proposal_id,
        "grammar.seed",
        None,
        value_type,
        representation,
        ("declaration.seed",),
    )


def descriptor(
    *, singularities: tuple[str, ...] = ("none",), proof_shape: tuple[str, ...] = ("direct",)
) -> D.BehaviorDescriptor:
    return D.BehaviorDescriptor(
        (0,),
        "projective",
        2,
        "rational",
        ("cross_ratio",),
        singularities,
        ("cross_ratio",),
        proof_shape,
    )


def test_four_extension_kinds_are_admitted_from_json_and_run_without_new_modules() -> None:
    verifiers, admissions = admission_registry()
    admitted = {
        item.declaration.kind: admissions.admit(item, verifiers.verify_all(item))
        for item in candidates()
    }
    assert set(admitted) == set(D.DeclarationKind)
    expression = seed("seed.expression", D.ValueType.EXPRESSION, "(a*x+b)/(c*x+d)")
    operator = admitted[D.DeclarationKind.OPERATOR]
    proposal = operator.emit_proposal(
        (expression,), parent_ids=(expression.proposal_id,), nonce="runtime-control"
    )
    assert proposal.declaration_id == "operator.mobius-normalize"
    assert proposal.representation == "mobius_normalize((a*x+b)/(c*x+d))"
    invariant = admitted[D.DeclarationKind.INVARIANT]
    scalars = tuple(seed(f"seed.{item}", D.ValueType.SCALAR, item) for item in "abcd")
    assert invariant.emit_proposal(
        scalars,
        parent_ids=tuple(item.proposal_id for item in scalars),
        nonce="invariant-control",
    ).representation == "cross_ratio(a,b,c,d)"
    assert admitted[D.DeclarationKind.GRAMMAR].execute(("Expr",)) == (
        "Atom",
        "quotient(Expr,nonzero(Expr))",
    )
    tactic = admitted[D.DeclarationKind.PROOF_TACTIC].as_tactic()
    assert tactic.mechanism == "case_split"


def test_extension_candidate_cannot_smuggle_decision_and_bad_replay_is_rejected() -> None:
    wire = candidates()[0].to_dict()
    with pytest.raises(R.RuntimeProtocolError, match="keys changed"):
        R.ExtensionCandidate.from_dict({**wire, "verdict": "verified"})
    malformed = {**wire, "capabilities": "not-a-json-array"}
    with pytest.raises(R.RuntimeProtocolError, match="JSON array"):
        R.ExtensionCandidate.from_dict(malformed)
    wire["tests"][0]["expected"] = "self-approved"
    candidate = R.ExtensionCandidate.from_dict(wire)
    verifiers, admissions = admission_registry()
    records = verifiers.verify_all(candidate)
    assert any(item.status is R.RuntimeStatus.REJECTED for item in records)
    assert any(item.blocker is not None for item in records)
    with pytest.raises(R.RuntimeProtocolError, match="rejection"):
        admissions.admit(candidate, records)


def test_extension_proposer_cannot_register_as_its_own_verifier() -> None:
    candidate = candidates()[0]
    registry = R.ExtensionVerifierRegistry()
    registry.register(
        identity("verifier.self", candidate.proposer_id, R.VerifierBackend.SCHEMA),
        R.structural_extension_verifier,
    )
    with pytest.raises(R.RuntimeProtocolError, match="own verifier"):
        registry.verify_all(candidate)


def test_extension_admission_rejects_a_reused_verifier_implementation() -> None:
    registry = R.ExtensionVerifierRegistry()
    registry.register(
        identity("verifier.first", "principal.first", R.VerifierBackend.SCHEMA),
        R.structural_extension_verifier,
    )
    with pytest.raises(R.RuntimeProtocolError, match="duplicate"):
        registry.register(
            identity(
                "verifier.renamed",
                "principal.second",
                R.VerifierBackend.EXACT_ARITHMETIC,
            ),
            R.structural_extension_verifier,
        )


def test_independent_result_quorum_covers_exact_cas_interval_and_lean() -> None:
    proposal = seed("proposal.quorum", D.ValueType.EXPRESSION, "x^2+2*x+1")
    policy = R.DecisionPolicy(
        (
            R.VerifierBackend.EXACT_ARITHMETIC,
            R.VerifierBackend.CAS,
            R.VerifierBackend.INTERVAL,
            R.VerifierBackend.LEAN,
        ),
        4,
    )
    registry = R.IndependentResultVerifierRegistry(policy)

    def verifier(candidate: D.Proposal, verifier_identity: R.VerifierIdentity) -> R.BackendDecision:
        record = D.VerificationRecord(
            candidate.proposal_id,
            verifier_identity.verifier_id,
            D.VerificationStatus.VERIFIED,
            Fraction(1),
            descriptor(),
        )
        return R.BackendDecision(
            verifier_identity,
            record,
            canonical_sha256(
                {"backend": verifier_identity.backend.value, "proposal": candidate.proposal_id}
            ),
        )

    for backend in policy.required_backends:
        registry.register(
            identity(
                f"verifier.{backend.value}",
                f"principal.{backend.value}",
                backend,
            ),
            verifier,
        )
    bundle = registry.decide(R.ProposedArtifact(proposal, "principal.generator"))
    assert bundle.verified
    assert {item.identity.backend for item in bundle.decisions} == set(policy.required_backends)
    assert len(bundle.bundle_sha256) == 64


def test_behavioral_archive_distinguishes_singularities_conservation_and_proof_shape() -> None:
    archive = D.BehavioralMapElites()
    proposals = (
        seed("proposal.behavior-a", D.ValueType.EXPRESSION, "x"),
        seed("proposal.behavior-b", D.ValueType.EXPRESSION, "x+0"),
        seed("proposal.behavior-c", D.ValueType.EXPRESSION, "1/x"),
    )
    descriptors = (
        descriptor(),
        descriptor(),
        descriptor(singularities=("simple_pole",), proof_shape=("residue",)),
    )
    qualities = (Fraction(1, 2), Fraction(3, 4), Fraction(2, 3))
    for proposal, behavior, quality in zip(proposals, descriptors, qualities, strict=True):
        archive.insert(
            proposal,
            D.VerificationRecord(
                proposal.proposal_id,
                "verifier.behavior",
                D.VerificationStatus.VERIFIED,
                quality,
                behavior,
            ),
        )
    assert archive.occupied_niches == 2
    assert {item.proposal_id for item in archive.elites()} == {
        "proposal.behavior-b",
        "proposal.behavior-c",
    }


def test_negative_publication_requires_feature_level_expressibility_and_verified_witness() -> None:
    transitions = (
        R.SearchTransition(
            "grammar.sequence-to-expression",
            D.ValueType.SEQUENCE,
            D.ValueType.EXPRESSION,
            ("generating_function",),
        ),
        R.SearchTransition(
            "grammar.expression-to-equation",
            D.ValueType.EXPRESSION,
            D.ValueType.EQUATION,
            ("holonomic", "recurrence"),
        ),
    )
    target = R.TargetContract(
        "target.holonomic-recurrence", D.ValueType.EQUATION, ("holonomic", "recurrence")
    )
    certificate = R.prove_expressibility(
        D.ValueType.SEQUENCE,
        target,
        transitions,
        witness_proposal_id="proposal.known-holonomic",
        witness_verification_sha256=canonical_sha256("verified witness"),
    )
    negative = R.publish_qualified_negative(
        4096, canonical_sha256("complete enumeration"), certificate, transitions
    )
    assert negative.status == "REACHABILITY_QUALIFIED_NEGATIVE"
    impossible = R.TargetContract(
        "target.singular-holonomic",
        D.ValueType.EQUATION,
        ("essential_singularity", "holonomic"),
    )
    with pytest.raises(R.RuntimeProtocolError, match="not expressible"):
        R.prove_expressibility(
            D.ValueType.SEQUENCE,
            impossible,
            transitions,
            witness_proposal_id="proposal.missing",
            witness_verification_sha256=canonical_sha256("missing"),
        )


def test_proof_plan_search_tracks_invariants_induction_normal_forms_and_blockers() -> None:
    tactics = (
        R.ProofTacticSpec(
            "tactic.induct",
            "recurrence",
            ("step",),
            (),
            ("monotone",),
            ("n",),
            "normalized_recurrence",
            "generating_function",
            "induction",
        ),
        R.ProofTacticSpec(
            "tactic.close-step",
            "step",
            (),
            ("monotone",),
            (),
            (),
            "coefficient_normal_form",
            "polynomial",
            "coefficient_extraction",
        ),
    )
    plan = R.search_operational_proof_plan(
        "proposal.recurrence", (R.ProofGoal("recurrence"),), tactics
    )
    assert plan.closed
    assert plan.tactic_ids == ("tactic.induct", "tactic.close-step")
    assert plan.mechanisms == ("coefficient_extraction", "induction")
    failed = R.search_operational_proof_plan(
        "proposal.unclosed", (R.ProofGoal("analytic_boundary"),), tactics
    )
    assert not failed.closed
    assert failed.blockers[0].kind is D.BlockerKind.PROOF_OBLIGATION
    assert failed.blockers[0].distance > 0


def test_dataset_pipeline_infers_groups_and_coordinates_before_fit_and_opens_holdout_last() -> None:
    reveal = json.dumps({"objects": [[5, 15], [6, 21]]}, sort_keys=True)
    pipeline = R.OperationalDatasetPipeline(
        canonical_sha256({"training": [[1, 1], [2, 3], [3, 6]]}),
        R.text_commitment(reveal),
    )
    previous = pipeline.dataset_sha256
    for index, stage in enumerate(R.DATASET_STAGES_V2):
        output = canonical_sha256({"input": previous, "stage": stage.value})
        pipeline.record(
            stage,
            previous,
            output,
            artifacts=(f"artifact.{stage.value}",),
            metric=Fraction(index, 10),
            passed=True,
            heldout_reveal=reveal if stage is R.DatasetStageV2.HELDOUT_TEST else None,
        )
        previous = output
    assert pipeline.completed
    stages = [item.stage for item in pipeline.records]
    assert stages.index(R.DatasetStageV2.DIMENSIONLESS_GROUPS) < stages.index(
        R.DatasetStageV2.SYMBOLIC_LAW_FIT
    )
    assert stages[-1] is R.DatasetStageV2.HELDOUT_TEST
    early = R.OperationalDatasetPipeline("d" * 64, R.text_commitment(reveal))
    with pytest.raises(R.RuntimeProtocolError, match="before the final"):
        early.record(
            R.DatasetStageV2.SHAPE_AUDIT,
            "d" * 64,
            "e" * 64,
            artifacts=("artifact.shape",),
            metric=Fraction(0),
            passed=True,
            heldout_reveal=reveal,
        )


def test_failed_dataset_stage_returns_typed_blocker_and_stops_pipeline() -> None:
    pipeline = R.OperationalDatasetPipeline("d" * 64, "e" * 64)
    blocker = D.TypedBlocker(
        "blocker.dataset.units",
        D.BlockerKind.DATA_INADEQUACY,
        D.ValueType.DATASET_MODEL,
        Fraction(1, 4),
        "missing unit column",
        D.CreativityOperator.COUNTEREXAMPLE_REPAIR,
    )
    pipeline.record(
        R.DatasetStageV2.SHAPE_AUDIT,
        "d" * 64,
        "f" * 64,
        artifacts=("artifact.shape",),
        metric=Fraction(1),
        passed=False,
        blocker=blocker,
    )
    with pytest.raises(R.RuntimeProtocolError, match="cannot continue"):
        pipeline.record(
            R.DatasetStageV2.UNIT_NORMALIZATION,
            "f" * 64,
            "a" * 64,
            artifacts=("artifact.units",),
            metric=Fraction(0),
            passed=True,
        )


def blind_result(level: R.CapabilityLevelV2, *, passed: bool = True) -> R.BlindBenchmarkResult:
    reveal = f"sealed-target-level-{level.value}"
    return R.BlindBenchmarkResult(
        level,
        f"benchmark.level-{level.value}",
        "principal.blind-generator",
        R.text_commitment(reveal) if level.value >= 3 else "",
        canonical_sha256({"proposal": level.value}),
        level.value * 2,
        reveal,
        level.value * 2 + 1,
        (),
        (R.VerifierBackend.EXACT_ARITHMETIC,),
        passed,
    )


def test_blind_ladder_cryptographically_opens_levels_and_gates_open_problem_spend() -> None:
    ladder = R.BlindCapabilityLadderV2()
    for level in tuple(R.CapabilityLevelV2)[:5]:
        ladder.admit(blind_result(level))
    ladder.admit(blind_result(R.CapabilityLevelV2.OPEN_PROBLEM, passed=False))
    assert ladder.highest_passed == 5
    assert ladder.open_problem_spend_authorized
    broken = blind_result(R.CapabilityLevelV2.SYNTHETIC_TARGET_SEALED)
    with pytest.raises(R.RuntimeProtocolError, match="reveal"):
        R.BlindBenchmarkResult(
            broken.level,
            broken.benchmark_id,
            broken.proposer_principal_id,
            "0" * 64,
            broken.proposal_sha256,
            broken.proposal_sequence,
            broken.target_reveal,
            broken.reveal_sequence,
            (),
            broken.verifier_backends,
            True,
        )


def test_serious_claim_release_requires_blind_holdout_reproduction_certificate_and_human_review() -> None:
    chain = R.SeriousClaimChain()
    rows = (
        R.EvidenceLink("d", R.EvidenceStage.DECLARATION, "1" * 64, "actor.declarer", (), "typed_declaration"),
        R.EvidenceLink("c", R.EvidenceStage.TARGET_COMMITMENT, "2" * 64, "actor.sealer", ("d",), "target_commitment"),
        R.EvidenceLink("p", R.EvidenceStage.BLIND_PROPOSAL, "3" * 64, "actor.proposer", ("c",), "blind_generation"),
        R.EvidenceLink("h", R.EvidenceStage.HOLDOUT_SURVIVAL, "4" * 64, "actor.holdout", ("p",), "sealed_holdout"),
        R.EvidenceLink("i", R.EvidenceStage.INDEPENDENT_REPRODUCTION, "5" * 64, "actor.reproducer", ("h",), "independent_replay"),
        R.EvidenceLink("e", R.EvidenceStage.EXACT_CERTIFICATE, "6" * 64, "actor.exact", ("i",), "exact_certificate"),
        R.EvidenceLink("a", R.EvidenceStage.HUMAN_PRIOR_ART, "7" * 64, "actor.human", ("e",), "human_review"),
        R.EvidenceLink("r", R.EvidenceStage.RELEASE, "8" * 64, "actor.release", ("a",), "claim_release"),
    )
    for row in rows:
        chain.add(row)
    assert len(chain.validate_release("r")) == 64
    incomplete = R.SeriousClaimChain()
    for row in rows[:5]:
        incomplete.add(row)
    incomplete.add(
        R.EvidenceLink(
            "r", R.EvidenceStage.RELEASE, "8" * 64, "actor.release", ("i",), "claim_release"
        )
    )
    with pytest.raises(R.RuntimeProtocolError, match="incomplete"):
        incomplete.validate_release("r")


def test_operational_metrics_report_every_requested_creative_yield_without_claiming_novelty() -> None:
    value = R.OperationalCreativeYield(
        proposals=100,
        verified=20,
        behavioral_niches=12,
        unique_proof_mechanisms=4,
        proof_plans_attempted=10,
        proof_plans_closed=3,
        counterexamples_tested=50,
        counterexample_survivors=5,
        holdout_baseline_loss=Fraction(3, 5),
        holdout_best_loss=Fraction(1, 5),
        positives=2,
        gpu_milliseconds_construction=3_600_000,
        gpu_milliseconds_refutation=1_800_000,
    ).to_dict()
    assert value["counts"]["behavioral_niches"] == 12
    assert value["counts"]["unique_proof_mechanisms"] == 4
    assert value["rates"]["holdout_improvement"] == "2/3"
    assert value["rates"]["proof_completion"] == "3/10"
    assert value["rates"]["counterexample_survival"] == "1/10"
    assert value["compute"]["gpu_hours_per_positive"] == "3/4"
    assert value["compute"]["construction_to_refutation"] == "2/1"
    assert value["claims"]["novelty_established"] is False
