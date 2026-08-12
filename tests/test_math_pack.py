from __future__ import annotations

from fractions import Fraction

from sigma_theory_compiler.math_expression_ir import (
    Equation,
    Inequality,
    InequalityRelation,
    literal,
    symbol,
)
from sigma_theory_compiler.math_pack import (
    MathDomainPack,
    math_candidate_representation,
    math_pack_descriptor,
)
from sigma_theory_compiler.math_types import IntegerType
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    OutcomeStatus,
    PromotionLedger,
    ProvenanceRecord,
    SourceBinding,
    run_gate,
    run_stage,
)

RECEIPT_HASH = "a" * 64
SOURCE_HASH = "b" * 64


def make_candidate(
    formula: Equation | Inequality,
    *,
    canonical_override: str | None = None,
    prior_art: bool = True,
) -> CandidateArtifact:
    descriptor = math_pack_descriptor()
    sources = [SourceBinding("generator", "benchmarks/generator.json", SOURCE_HASH)]
    receipt = RECEIPT_HASH if prior_art else None
    if prior_art:
        sources.append(SourceBinding("prior_art_receipt", "evidence/prior-art.json", RECEIPT_HASH))
    representation = math_candidate_representation(
        formula,
        {"n": IntegerType(0, 100)},
        exact_assignments=({"n": 0}, {"n": 1}, {"n": 10}, {"n": 100}),
        random_trials=16,
        adversarial_limit=16,
        seed=17,
        prior_art_receipt_sha256=receipt,
    )
    if canonical_override is not None:
        representation["canonical_formula_sha256"] = canonical_override
    provenance = ProvenanceRecord.create(
        descriptor.ref,
        {"benchmark": "math-pack-adapter", "seed": 17},
        sources=sources,
    )
    return CandidateArtifact.create(
        ArtifactKind.IDENTITY,
        "The two exact expressions are identical for every declared integer.",
        representation,
        provenance,
        assumptions=("n is an integer in the declared test domain",),
        claims=("exact_identity",),
    )


def exact_identity() -> Equation:
    n = symbol("n")
    return Equation(n * (n + 1) / 2, (n**2 + n) / 2)


def run_to_stage(
    pack: MathDomainPack,
    artifact: CandidateArtifact,
    final_stage: str,
) -> dict[str, object]:
    outcomes = {}
    for stage in pack.descriptor.stages:
        outcome = run_stage(pack, artifact, stage.stage_id, outcomes)
        outcomes[stage.stage_id] = outcome
        if stage.stage_id == final_stage or outcome.status is not OutcomeStatus.PASS:
            break
    return outcomes


def test_descriptor_is_closed_and_domain_independent() -> None:
    descriptor = math_pack_descriptor()
    assert descriptor.pack_id == "sigma.math"
    assert tuple(stage.stage_id for stage in descriptor.stages) == (
        "typed",
        "canonicalized",
        "counterexample_screened",
        "exactly_verified",
        "prior_art_checked",
    )
    assert tuple(gate.gate_id for gate in descriptor.gates) == tuple(
        sorted(gate.gate_id for gate in descriptor.gates)
    )
    assert ArtifactKind.PHYSICAL_ACTION not in descriptor.supported_kinds


def test_exact_identity_passes_every_stage_and_hash_chained_promotion() -> None:
    pack = MathDomainPack()
    artifact = make_candidate(exact_identity())
    outcomes = run_to_stage(pack, artifact, "prior_art_checked")
    assert all(outcome.status is OutcomeStatus.PASS for outcome in outcomes.values())

    ledger = PromotionLedger.create(artifact)
    for stage in pack.descriptor.stages:
        required = {
            name: outcomes[name]
            for name in pack.descriptor.gate(f"admit_{stage.stage_id}").required_stages
        }
        gate = run_gate(pack, artifact, f"admit_{stage.stage_id}", required)
        assert gate.status is OutcomeStatus.PASS
        ledger = ledger.promote(pack.descriptor, artifact, gate, required)
    assert ledger.current_stage == "prior_art_checked"
    assert len(ledger.entries) == 5
    assert ledger.entries[-1].prior_entry_sha256 == ledger.entries[-2].entry_sha256


def test_wrong_formula_is_rejected_by_counterexample_search() -> None:
    n = symbol("n")
    artifact = make_candidate(Equation(n * (n + 1) / 2, n**2))
    outcomes = run_to_stage(MathDomainPack(), artifact, "counterexample_screened")
    assert outcomes["typed"].status is OutcomeStatus.PASS
    assert outcomes["canonicalized"].status is OutcomeStatus.PASS
    assert outcomes["counterexample_screened"].status is OutcomeStatus.REJECT
    assert outcomes["counterexample_screened"].reason_codes == ("counterexample_found",)


def test_bounded_counterexample_exhaustion_does_not_become_proof() -> None:
    n = symbol("n")
    formula = Inequality(n**2, literal(0), InequalityRelation.GREATER_EQUAL)
    artifact = make_candidate(formula)
    outcomes = run_to_stage(MathDomainPack(), artifact, "exactly_verified")
    assert outcomes["counterexample_screened"].status is OutcomeStatus.PASS
    assert outcomes["exactly_verified"].status is OutcomeStatus.BLOCK
    assert outcomes["exactly_verified"].reason_codes == ("exact_proof_not_closed",)


def test_canonical_hash_tamper_rejects_before_expensive_checks() -> None:
    artifact = make_candidate(exact_identity(), canonical_override="0" * 64)
    outcomes = run_to_stage(MathDomainPack(), artifact, "canonicalized")
    assert outcomes["typed"].status is OutcomeStatus.PASS
    assert outcomes["canonicalized"].status is OutcomeStatus.REJECT


def test_prior_art_comparison_requires_a_bound_post_proof_receipt() -> None:
    artifact = make_candidate(exact_identity(), prior_art=False)
    outcomes = run_to_stage(MathDomainPack(), artifact, "prior_art_checked")
    assert outcomes["exactly_verified"].status is OutcomeStatus.PASS
    assert outcomes["prior_art_checked"].status is OutcomeStatus.BLOCK
    assert outcomes["prior_art_checked"].reason_codes == ("prior_art_receipt_missing_or_unbound",)


def test_undeclared_symbol_is_rejected_by_type_stage() -> None:
    n = symbol("n")
    m = symbol("m")
    artifact = make_candidate(Equation(n + m, m + n))
    outcome = run_stage(MathDomainPack(), artifact, "typed")
    assert outcome.status is OutcomeStatus.REJECT
    assert outcome.reason_codes == ("undeclared_symbols",)


def test_malformed_domain_representation_fails_closed_as_error() -> None:
    artifact = make_candidate(exact_identity())
    value = artifact.to_dict()
    value["representation"]["unregistered"] = True
    from sigma_theory_compiler.sigma_core import canonical_sha256

    body = {
        key: item for key, item in value.items() if key not in {"artifact_id", "content_sha256"}
    }
    digest = canonical_sha256(body)
    value["content_sha256"] = digest
    value["artifact_id"] = f"sig-{digest[:24]}"
    tampered = CandidateArtifact.from_dict(value)
    outcome = run_stage(MathDomainPack(), tampered, "typed")
    assert outcome.status is OutcomeStatus.ERROR
    assert outcome.reason_codes == ("domain_pack_error",)


def test_rational_assignments_round_trip_through_exact_candidate_boundary() -> None:
    n = symbol("n")
    representation = math_candidate_representation(
        Equation(n / 2 + n / 2, n),
        {"n": IntegerType(-10, 10)},
        exact_assignments=({"n": Fraction(1, 1)},),
        prior_art_receipt_sha256=RECEIPT_HASH,
    )
    encoded = representation["counterexample_plan"]["exact_assignments"][0]["n"]
    assert encoded == {"numerator": 1, "denominator": 1}
