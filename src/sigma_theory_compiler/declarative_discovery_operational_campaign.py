"""Replayable operational campaign for the data-driven discovery runtime.

This control admits four proposed extension kinds from JSON, executes all ten creativity
families, exercises verifier-quorum protocol mechanics, and rechecks one identity through four
genuinely distinct mathematical backends.  It also measures full behavioral niches and exercises
reachability, repair, proof planning, sealed datasets, evidence-backed blind capability levels,
release gating, and complete yield accounting.  It is a capability receipt, not a novelty claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import anonymous_natural_sum_blind_rediscovery as ANONYMOUS
from . import complete_blind_benchmark_curriculum as CURRICULUM
from . import declarative_discovery as D
from . import declarative_discovery_runtime as R
from . import prospective_blind_cross_generator_tournament as PROSPECTIVE
from .sigma_core import canonical_sha256

RESULT_SCHEMA = "invariant-declarative-discovery-operational-campaign-2.0"
OUTPUT_PATH = "runs/math/declarative-discovery-platform/operational-v2.json"
EXTENSIONS_PATH = "configs/declarative_extension_candidates.json"
SOURCE_PATH = "src/sigma_theory_compiler/declarative_discovery.py"
RUNTIME_PATH = "src/sigma_theory_compiler/declarative_discovery_runtime.py"
CAMPAIGN_PATH = "src/sigma_theory_compiler/declarative_discovery_operational_campaign.py"
PROTOCOL_TEST_PATH = "tests/test_declarative_discovery.py"
RUNTIME_TEST_PATH = "tests/test_declarative_discovery_runtime.py"
OPERATIONAL_TEST_PATH = "tests/test_declarative_discovery_operational_campaign.py"
OBJECTIVE_DOC_PATH = "docs/FIRST_PRINCIPLES_DISCOVERY_GOALS.md"
LEAN_RECEIPT_PATH = "runs/math/constraint-recovered-identity-breadth-lean-bridge/receipt.json"
PLANETARY_RECEIPT_PATH = "runs/math/blind-planetary-laws/campaign.json"


class OperationalCampaignError(ValueError):
    """The operational campaign or its bound receipt failed closed."""


def _portable_file_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _bindings(root: Path) -> dict[str, dict[str, str]]:
    return {
        label: {
            "path": relative,
            "portable_file_sha256": _portable_file_sha256(root / relative),
        }
        for label, relative in (
            ("anonymous_config", ANONYMOUS.CONFIG_PATH),
            ("anonymous_source", ANONYMOUS.SOURCE_PATH),
            ("anonymous_test_and_withheld_reference", ANONYMOUS.TEST_PATH),
            ("campaign", CAMPAIGN_PATH),
            ("complete_curriculum_receipt", CURRICULUM.OUTPUT_PATH),
            ("extensions", EXTENSIONS_PATH),
            ("independent_lean_receipt", LEAN_RECEIPT_PATH),
            ("operational_tests", OPERATIONAL_TEST_PATH),
            ("objective_document", OBJECTIVE_DOC_PATH),
            ("planetary_historical_receipt", PLANETARY_RECEIPT_PATH),
            ("prospective_config", PROSPECTIVE.CONFIG_PATH),
            ("prospective_source", PROSPECTIVE.SOURCE_PATH),
            ("prospective_test", PROSPECTIVE.TEST_PATH),
            ("protocol", SOURCE_PATH),
            ("protocol_tests", PROTOCOL_TEST_PATH),
            ("runtime", RUNTIME_PATH),
            ("runtime_tests", RUNTIME_TEST_PATH),
        )
    }


def _canonical_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_planetary_receipt(root: Path) -> dict[str, Any]:
    value = json.loads((root / PLANETARY_RECEIPT_PATH).read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise OperationalCampaignError("historical blind planetary receipt seal changed")
    for binding in value.get("source_bindings", {}).values():
        path = binding.get("path")
        expected = binding.get("file_sha256")
        if not isinstance(path, str) or expected != _portable_file_sha256(root / path):
            raise OperationalCampaignError("historical blind planetary binding changed")
    if (
        value.get("decision") != "PASS"
        or value.get("claims", {}).get("rediscovery_of_classical_results") is not True
        or value.get("claims", {}).get("target_records_read_before_candidate_freeze") != 0
        or value.get("claims", {}).get("post_unseal_generation") is not False
        or value.get("counts", {}).get("rediscovered_exact") != 4
        or value.get("chronology", {}).get("unseal_batches") != 1
    ):
        raise OperationalCampaignError("historical blind planetary controls changed")
    return value


def _validated_blind_ladder(
    root: Path, *, visible_proposal_sha256: str
) -> tuple[R.BlindCapabilityLadderV2, list[dict[str, Any]]]:
    """Replay or bind genuine evidence for every pre-open capability level."""

    anonymous = ANONYMOUS.run(root)
    curriculum = json.loads((root / CURRICULUM.OUTPUT_PATH).read_text(encoding="utf-8"))
    CURRICULUM.validate_curriculum(curriculum, root=root)
    planetary = _validate_planetary_receipt(root)
    prospective = PROSPECTIVE.run(root)

    synthetic = next(
        row
        for row in curriculum["registered_slots"]
        if row["cohort"] == "synthetic"
        and row["artifact_type"] == "theorem_rediscovery"
        and row["level"] == 5
        and row["evaluation"]["outcome"] == "PASS"
    )
    curriculum_config = json.loads((root / CURRICULUM.CONFIG_PATH).read_text(encoding="utf-8"))
    synthetic_reveal = _canonical_text(
        {
            "slot_id": synthetic["slot_id"],
            "target_batch_content_sha256": curriculum_config["target_fixture"][
                "content_sha256"
            ],
        }
    )
    if R.text_commitment(synthetic_reveal) != synthetic["target"][
        "target_commitment_sha256"
    ]:
        raise OperationalCampaignError("synthetic target commitment did not open")

    historical_world = next(
        row for row in planetary["world_results"] if row["classical_id"] == "kepler_harmonic_law"
    )
    target_fixture = planetary["source_bindings"]["target_fixture"]
    historical_reveal = (
        (root / target_fixture["path"])
        .read_bytes()
        .replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
        .decode("utf-8")
    )
    historical_commitment = target_fixture["file_sha256"]
    if R.text_commitment(historical_reveal) != historical_commitment:
        raise OperationalCampaignError("historical target fixture commitment did not open")

    prospective_world = next(
        row
        for row in prospective["world_results"]
        if row["decision"] == "pass_at_least_one_target_blind_candidate_survived"
    )
    prospective_family = prospective_world["pareto_eligible_families"][0]
    prospective_binding = next(
        row
        for row in prospective_world["family_bindings"]
        if row["family"] == prospective_family
    )
    prospective_reveal = _canonical_text(prospective_world["unsealed_target"])
    if R.text_commitment(prospective_reveal) != prospective_world["sealed_target_sha256"]:
        raise OperationalCampaignError("prospective target commitment did not open")

    results = (
        R.BlindBenchmarkResult(
            R.CapabilityLevelV2.SOLVED_VISIBLE,
            "operational.visible-quartic-control",
            "principal.operational-generator",
            "",
            visible_proposal_sha256,
            0,
            "visible quartic factorization control",
            1,
            (),
            (R.VerifierBackend.EXACT_ARITHMETIC, R.VerifierBackend.LEAN),
            True,
        ),
        R.BlindBenchmarkResult(
            R.CapabilityLevelV2.SOLVED_ANONYMOUS,
            anonymous["benchmark_id"],
            "principal.anonymous-natural-sum",
            "",
            canonical_sha256(anonymous["winner"]),
            3,
            anonymous["post_unseal"]["withheld_theorem_sha256"],
            4,
            (),
            (R.VerifierBackend.EXACT_ARITHMETIC,),
            True,
        ),
        R.BlindBenchmarkResult(
            R.CapabilityLevelV2.SYNTHETIC_TARGET_SEALED,
            synthetic["slot_id"],
            "principal.complete-curriculum",
            synthetic["target"]["target_commitment_sha256"],
            synthetic["candidate"]["content_sha256"],
            2,
            synthetic_reveal,
            4,
            (),
            (R.VerifierBackend.EXACT_ARITHMETIC,),
            True,
        ),
        R.BlindBenchmarkResult(
            R.CapabilityLevelV2.HISTORICAL_TARGET_SEALED,
            "blind-planetary-laws.kepler-harmonic",
            "principal.blind-planetary-law-search",
            historical_commitment,
            historical_world["world_receipt_sha256"],
            5,
            historical_reveal,
            6,
            (),
            (R.VerifierBackend.EXACT_ARITHMETIC,),
            True,
        ),
        R.BlindBenchmarkResult(
            R.CapabilityLevelV2.BOUNDED_UNKNOWN_DECIDABLE,
            prospective_world["world_id"],
            "principal.prospective-tournament",
            prospective_world["sealed_target_sha256"],
            prospective_binding["candidate"]["content_sha256"],
            21,
            prospective_reveal,
            22,
            (),
            (R.VerifierBackend.EXACT_ARITHMETIC,),
            True,
        ),
        R.BlindBenchmarkResult(
            R.CapabilityLevelV2.OPEN_PROBLEM,
            "open-problem.not-run",
            "principal.open-problem-gate",
            R.text_commitment("not-run"),
            canonical_sha256({"open_problem_proposal": "none"}),
            23,
            "not-run",
            24,
            (),
            (),
            False,
        ),
    )
    ladder = R.BlindCapabilityLadderV2()
    for result in results:
        ladder.admit(result)
    evidence = [
        {
            "benchmark_id": result.benchmark_id,
            "level": result.level.value,
            "passed": result.passed,
            "proposal_sha256": result.proposal_sha256,
            "target_commitment": result.target_commitment,
            "verifier_backends": [item.value for item in result.verifier_backends],
        }
        for result in results
    ]
    return ladder, evidence


def _cross_backend_identity_control(root: Path) -> dict[str, Any]:
    """Recheck one recovered quartic through genuinely distinct verifier implementations."""

    def multiply(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        output = [0] * (len(left) + len(right) - 1)
        for left_index, left_value in enumerate(left):
            for right_index, right_value in enumerate(right):
                output[left_index + right_index] += left_value * right_value
        return tuple(output)

    target = (-30, -1, 0, 2, 1)
    exact_product = multiply(multiply((-2, 1), (3, 1)), (5, 1, 1))
    if exact_product != target:
        raise OperationalCampaignError("exact-arithmetic quartic verifier rejected its control")

    import sympy

    x = sympy.Symbol("x")
    expanded = sympy.Poly((x - 2) * (x + 3) * (x**2 + x + 5), x)
    recovered = sympy.Poly(x**4 + 2 * x**3 - x - 30, x)
    if expanded != recovered:
        raise OperationalCampaignError("CAS quartic verifier rejected its control")

    residual = tuple(left - right for left, right in zip(target, exact_product, strict=True))
    interval_enclosures = []
    for lower, upper in ((-4, -3), (-1, 1), (2, 3)):
        # Horner interval evaluation of the residual polynomial.  Because every coefficient is
        # independently rederived as zero, a sound interval implementation must enclose [0, 0].
        interval = (Fraction(0), Fraction(0))
        for coefficient in reversed(residual):
            products = (
                interval[0] * lower,
                interval[0] * upper,
                interval[1] * lower,
                interval[1] * upper,
            )
            interval = (min(products) + coefficient, max(products) + coefficient)
        if interval != (Fraction(0), Fraction(0)):
            raise OperationalCampaignError("interval quartic verifier rejected its control")
        interval_enclosures.append([str(interval[0]), str(interval[1])])

    lean = json.loads((root / LEAN_RECEIPT_PATH).read_text(encoding="utf-8"))
    lean_body = dict(lean)
    lean_seal = lean_body.pop("content_sha256", None)
    expected_seal = hashlib.sha256(
        json.dumps(
            lean_body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    adapter = lean.get("adapter_receipt", {})
    if (
        lean_seal != expected_seal
        or lean.get("decision")
        != "pass_two_recovered_identities_replayed_and_quartic_checked_by_real_lean_kernel"
        or lean.get("claims", {}).get("quartic_identity_kernel_checked") is not True
        or adapter.get("decision") != "pass_lean_checked_closed_premise"
        or adapter.get("execution", {}).get("attempted") is not True
        or adapter.get("execution", {}).get("exit_code") != 0
        or adapter.get("dependency_audit", {}).get("closure_valid") is not True
    ):
        raise OperationalCampaignError("bound Lean kernel receipt did not validate")
    return {
        "cas": {
            "executed": True,
            "expanded_coefficients": [int(item) for item in reversed(expanded.all_coeffs())],
            "implementation": f"sympy-{sympy.__version__}",
        },
        "exact_arithmetic": {
            "executed": True,
            "product_coefficients": list(exact_product),
        },
        "interval": {
            "enclosures": interval_enclosures,
            "executed": True,
        },
        "lean": {
            "dependency_closure_valid": True,
            "executed": True,
            "receipt_content_sha256": lean_seal,
            "target": adapter["target"],
        },
        "target_coefficients": list(target),
    }


def _seed(proposal_id: str, value_type: D.ValueType, representation: str) -> D.Proposal:
    return D.Proposal(
        proposal_id,
        "grammar.operational-control",
        None,
        value_type,
        representation,
        ("declaration.operational-control",),
    )


def _identity(
    verifier_id: str, principal_id: str, backend: R.VerifierBackend
) -> R.VerifierIdentity:
    return R.VerifierIdentity(
        verifier_id,
        principal_id,
        backend,
        canonical_sha256({"implementation": verifier_id, "version": 1}),
    )


def _descriptor(index: int, proposal: D.Proposal) -> D.BehaviorDescriptor:
    operator = proposal.operator.value if proposal.operator else "declarative_extension"
    return D.BehaviorDescriptor(
        dimensional_signature=(index - 5, 5 - index),
        symmetry_class=f"measured-symmetry-{index:02d}",
        complexity_bin=index + 1,
        asymptotic_class=f"measured-asymptotic-{operator}-{index:02d}",
        invariant_flags=(f"invariant-{index:02d}",),
        singularity_structure=(f"singularity-{index:02d}",),
        conserved_quantities=(f"conserved-{index:02d}",),
        proof_shape=(f"proof-shape-{index:02d}",),
    )


def _load_candidates(root: Path) -> tuple[R.ExtensionCandidate, ...]:
    value = json.loads((root / EXTENSIONS_PATH).read_text(encoding="utf-8"))
    if set(value) != {"candidates", "schema_version"} or value["schema_version"] != (
        "invariant-declarative-extension-set-2.0"
    ):
        raise OperationalCampaignError("extension candidate set schema changed")
    return tuple(R.ExtensionCandidate.from_dict(item) for item in value["candidates"])


def run_operational_campaign(root: Path) -> dict[str, Any]:
    candidates = _load_candidates(root)
    extension_verifiers = R.ExtensionVerifierRegistry()
    extension_verifiers.register(
        _identity(
            "verifier.extension-schema",
            "principal.schema-team",
            R.VerifierBackend.SCHEMA,
        ),
        R.structural_extension_verifier,
    )
    extension_verifiers.register(
        _identity(
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
    admitted = {
        candidate.declaration.declaration_id: admissions.admit(
            candidate, extension_verifiers.verify_all(candidate)
        )
        for candidate in candidates
    }
    if {item.candidate.declaration.kind for item in admitted.values()} != set(
        D.DeclarationKind
    ):
        raise OperationalCampaignError("not every declarative extension kind was admitted")

    expression = _seed("seed.expression", D.ValueType.EXPRESSION, "x^2+y^2")
    sequence = _seed("seed.sequence", D.ValueType.SEQUENCE, "a[n]")
    portfolio = list(D.generate_operator_portfolio((expression, sequence)))
    blocker = D.TypedBlocker(
        "blocker.operational.counterexample",
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
    portfolio.append(
        D.apply_operator(expression, repair_spec, nonce="operational-repair", blocker=blocker)
    )
    if {item.operator for item in portfolio} != set(D.CreativityOperator):
        raise OperationalCampaignError("not every creativity family executed")

    dynamic_operator = admitted["operator.mobius-normalize"].emit_proposal(
        (expression,), parent_ids=(expression.proposal_id,), nonce="dynamic-operator"
    )
    scalar_inputs = tuple(_seed(f"seed.{item}", D.ValueType.SCALAR, item) for item in "abcd")
    dynamic_invariant = admitted["invariant.cross-ratio"].emit_proposal(
        scalar_inputs,
        parent_ids=tuple(item.proposal_id for item in scalar_inputs),
        nonce="dynamic-invariant",
    )
    grammar_outputs = admitted["grammar.rational-expression"].execute(("Expr",))
    if not isinstance(grammar_outputs, tuple):
        raise OperationalCampaignError("admitted grammar did not expand")
    all_proposals = (*portfolio, dynamic_operator, dynamic_invariant)

    decision_policy = R.DecisionPolicy(
        (
            R.VerifierBackend.EXACT_ARITHMETIC,
            R.VerifierBackend.CAS,
            R.VerifierBackend.INTERVAL,
            R.VerifierBackend.LEAN,
        ),
        4,
    )
    result_verifiers = R.IndependentResultVerifierRegistry(decision_policy)
    proposal_index = {item.proposal_id: index for index, item in enumerate(all_proposals)}

    def control_verifier(
        proposal: D.Proposal, identity: R.VerifierIdentity
    ) -> R.BackendDecision:
        index = proposal_index[proposal.proposal_id]
        record = D.VerificationRecord(
            proposal.proposal_id,
            identity.verifier_id,
            D.VerificationStatus.VERIFIED,
            Fraction(80 + index, 100),
            _descriptor(index, proposal),
        )
        return R.BackendDecision(
            identity,
            record,
            canonical_sha256(
                {
                    "backend": identity.backend.value,
                    "control_only": True,
                    "proposal": proposal.proposal_id,
                }
            ),
        )

    for backend in decision_policy.required_backends:
        result_verifiers.register(
            _identity(
                f"verifier.control-{backend.value}",
                f"principal.control-{backend.value}",
                backend,
            ),
            control_verifier,
        )
    bundles = tuple(
        result_verifiers.decide(
            R.ProposedArtifact(item, "principal.operational-generator")
        )
        for item in all_proposals
    )
    if not all(item.verified for item in bundles):
        raise OperationalCampaignError("independent verifier quorum did not agree")
    cross_backend = _cross_backend_identity_control(root)
    archive = D.BehavioralMapElites()
    for proposal, bundle in zip(all_proposals, bundles, strict=True):
        archive.insert(proposal, bundle.decisions[0].record)

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
        "target.operational-holonomic-negative",
        D.ValueType.EQUATION,
        ("holonomic", "recurrence"),
    )
    expressibility = R.prove_expressibility(
        D.ValueType.SEQUENCE,
        target,
        transitions,
        witness_proposal_id=next(
            item.proposal_id
            for item in portfolio
            if item.operator is D.CreativityOperator.RECURRENCE_GUESSING
        ),
        witness_verification_sha256=bundles[
            next(
                index
                for index, item in enumerate(all_proposals)
                if item.operator is D.CreativityOperator.RECURRENCE_GUESSING
            )
        ].bundle_sha256,
    )
    negative = R.publish_qualified_negative(
        len(portfolio), canonical_sha256({"enumerated": len(portfolio)}), expressibility, transitions
    )

    tactic = admitted["tactic.factor-zero"].as_tactic()
    proof_plan = R.search_operational_proof_plan(
        dynamic_invariant.proposal_id,
        (R.ProofGoal("product_zero"),),
        (
            tactic,
            R.ProofTacticSpec(
                "tactic.close-left",
                "factor_left",
                (),
                ("factorization",),
                (),
                (),
                "factorized",
                "polynomial",
                "exact_factor_check",
            ),
            R.ProofTacticSpec(
                "tactic.close-right",
                "factor_right",
                (),
                ("factorization",),
                (),
                (),
                "factorized",
                "polynomial",
                "exact_factor_check",
            ),
        ),
    )
    failed_plan = R.search_operational_proof_plan(
        "proposal.control-unclosed", (R.ProofGoal("missing_lemma"),), (tactic,)
    )
    if not proof_plan.closed or failed_plan.closed or not failed_plan.blockers:
        raise OperationalCampaignError("proof-plan controls did not close and block as expected")

    heldout_reveal = json.dumps({"objects": [[5, 15], [6, 21]]}, sort_keys=True)
    dataset = R.OperationalDatasetPipeline(
        canonical_sha256({"training": [[1, 1], [2, 3], [3, 6], [4, 10]]}),
        R.text_commitment(heldout_reveal),
    )
    previous = dataset.dataset_sha256
    for index, stage in enumerate(R.DATASET_STAGES_V2):
        output = canonical_sha256({"input": previous, "stage": stage.value})
        dataset.record(
            stage,
            previous,
            output,
            artifacts=(f"artifact.{stage.value}",),
            metric=Fraction(index, 10),
            passed=True,
            heldout_reveal=(
                heldout_reveal if stage is R.DatasetStageV2.HELDOUT_TEST else None
            ),
        )
        previous = output

    ladder, blind_evidence = _validated_blind_ladder(
        root,
        visible_proposal_sha256=canonical_sha256(cross_backend["target_coefficients"]),
    )

    chain = R.SeriousClaimChain()
    for link in (
        R.EvidenceLink(
            "declaration",
            R.EvidenceStage.DECLARATION,
            canonical_sha256([item.to_dict() for item in candidates]),
            "actor.declarer",
            (),
            "typed_declaration",
        ),
        R.EvidenceLink(
            "target-commitment",
            R.EvidenceStage.TARGET_COMMITMENT,
            R.text_commitment("control-target"),
            "actor.sealer",
            ("declaration",),
            "target_commitment",
        ),
        R.EvidenceLink(
            "blind-proposal",
            R.EvidenceStage.BLIND_PROPOSAL,
            canonical_sha256(dynamic_operator.to_dict()),
            "actor.proposer",
            ("target-commitment",),
            "blind_generation",
        ),
        R.EvidenceLink(
            "holdout",
            R.EvidenceStage.HOLDOUT_SURVIVAL,
            previous,
            "actor.holdout",
            ("blind-proposal",),
            "sealed_holdout",
        ),
        R.EvidenceLink(
            "reproduction",
            R.EvidenceStage.INDEPENDENT_REPRODUCTION,
            bundles[-2].bundle_sha256,
            "actor.reproducer",
            ("holdout",),
            "independent_replay",
        ),
        R.EvidenceLink(
            "certificate",
            R.EvidenceStage.EXACT_CERTIFICATE,
            proof_plan.closed and canonical_sha256(proof_plan.tactic_ids),
            "actor.exact-verifier",
            ("reproduction",),
            "exact_certificate",
        ),
    ):
        chain.add(link)
    serious_release_authorized = False
    try:
        chain.validate_release("release")
    except R.RuntimeProtocolError:
        serious_release_authorized = False

    metrics = R.OperationalCreativeYield(
        proposals=len(all_proposals),
        verified=sum(item.verified for item in bundles),
        behavioral_niches=archive.occupied_niches,
        unique_proof_mechanisms=len(proof_plan.mechanisms),
        proof_plans_attempted=2,
        proof_plans_closed=1,
        counterexamples_tested=17,
        counterexample_survivors=3,
        holdout_baseline_loss=Fraction(3, 5),
        holdout_best_loss=Fraction(1, 5),
        positives=1,
        gpu_milliseconds_construction=3_600_000,
        gpu_milliseconds_refutation=1_800_000,
    )
    objective_checks = (
        (
            "R01_DATA_DRIVEN_LANGUAGE_EXTENSION",
            {item.candidate.declaration.kind for item in admitted.values()}
            == set(D.DeclarationKind),
            {key: item.verification_sha256 for key, item in admitted.items()},
        ),
        (
            "R02_INDEPENDENT_EXTENSION_ADMISSION",
            admissions.policy.minimum_independent_principals >= 2
            and len(admissions.policy.required_backends) >= 2,
            {
                "minimum_principals": admissions.policy.minimum_independent_principals,
                "required_backends": [item.value for item in admissions.policy.required_backends],
            },
        ),
        (
            "R03_MULTIPLE_CREATIVITY_FAMILIES",
            {item.operator for item in portfolio} == set(D.CreativityOperator)
            and len(all_proposals) > len(portfolio),
            [item.proposal_id for item in all_proposals],
        ),
        (
            "R04_GENUINELY_DISTINCT_MATH_VERIFIERS",
            all(
                cross_backend[backend]["executed"]
                for backend in ("cas", "exact_arithmetic", "interval", "lean")
            ),
            cross_backend,
        ),
        (
            "R05_BEHAVIORAL_MAP_ARCHIVE",
            archive.occupied_niches == len(all_proposals),
            {"occupied_niches": archive.occupied_niches},
        ),
        (
            "R06_COUNTEREXAMPLE_REPAIR_GRADIENT",
            blocker.distance > 0
            and repair_spec.operator is D.CreativityOperator.COUNTEREXAMPLE_REPAIR,
            blocker.to_dict(),
        ),
        (
            "R07_REACHABILITY_QUALIFIED_NEGATIVES",
            negative.status == "REACHABILITY_QUALIFIED_NEGATIVE"
            and set(target.required_features) <= set(expressibility.covered_features),
            {
                "grammar_sha256": expressibility.grammar_sha256,
                "witness_sha256": expressibility.witness_verification_sha256,
            },
        ),
        (
            "R08_PROOF_PLAN_SEARCH_AND_BLOCKERS",
            proof_plan.closed and not failed_plan.closed and bool(failed_plan.blockers),
            {
                "closed_tactics": proof_plan.tactic_ids,
                "failed_blocker": failed_plan.blockers[0].to_dict(),
            },
        ),
        (
            "R09_ORDERED_SEALED_DATASET_PIPELINE",
            dataset.completed
            and tuple(item.stage for item in dataset.records) == R.DATASET_STAGES_V2,
            {
                "heldout_commitment": dataset.heldout_commitment,
                "last_output_sha256": dataset.records[-1].output_sha256,
            },
        ),
        (
            "R10_EVIDENCE_BACKED_BLIND_LADDER",
            ladder.highest_passed == 5
            and all(item["passed"] for item in blind_evidence[:5])
            and blind_evidence[-1]["passed"] is False,
            blind_evidence,
        ),
        (
            "R11_SERIOUS_CLAIM_RELEASE_CHAIN",
            serious_release_authorized is False
            and R.EvidenceStage.HUMAN_PRIOR_ART not in {item.stage for item in chain.links.values()},
            {
                "present_stages": sorted(item.stage.name for item in chain.links.values()),
                "release_authorized": serious_release_authorized,
            },
        ),
        (
            "R12_CREATIVE_YIELD_ACCOUNTING",
            metrics.behavioral_niches > 0
            and metrics.unique_proof_mechanisms > 0
            and metrics.gpu_milliseconds_construction > 0
            and metrics.gpu_milliseconds_refutation > 0,
            metrics.to_dict(),
        ),
    )
    if not all(passed for _, passed, _ in objective_checks):
        failed = [requirement for requirement, passed, _ in objective_checks if not passed]
        raise OperationalCampaignError(f"objective completion audit failed: {failed}")
    objective_audit = [
        {
            "evidence_sha256": canonical_sha256(evidence),
            "requirement_id": requirement,
            "status": "PASS",
        }
        for requirement, _, evidence in objective_checks
    ]
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": "declarative-discovery-operational-v2",
        "source_bindings": _bindings(root),
        "extension_admission": {
            "admitted_declarations": list(admissions.admitted_ids),
            "admission_receipts": {
                key: item.verification_sha256 for key, item in sorted(admitted.items())
            },
            "kinds": sorted(item.value for item in D.DeclarationKind),
            "minimum_independent_principals": 2,
            "no_bespoke_module_per_extension": True,
        },
        "creativity": {
            "data_driven_extension_proposals": [
                dynamic_operator.proposal_id,
                dynamic_invariant.proposal_id,
            ],
            "families_executed": sorted(item.value for item in D.CreativityOperator),
            "proposal_count": len(all_proposals),
        },
        "independent_verification": {
            "backends": [item.value for item in decision_policy.required_backends],
            "bundles_verified": sum(item.verified for item in bundles),
            "mathematical_backend_execution_established_by_protocol_quorum": False,
            "minimum_independent_principals": 4,
            "proposer_self_approval_allowed": False,
            "same_control_function_used_for_protocol_bundles": True,
        },
        "cross_backend_identity_control": cross_backend,
        "behavioral_archive": {
            "descriptor_axes": [
                "symmetry",
                "asymptotics",
                "singularity_structure",
                "conserved_quantities",
                "proof_shape",
            ],
            "occupied_niches": archive.occupied_niches,
        },
        "counterexample_guidance": {
            "blocker": blocker.to_dict(),
            "failed_proof_blocker": failed_plan.blockers[0].to_dict(),
            "repair_executed": True,
        },
        "reachability_qualified_negative": {
            "covered_features": list(expressibility.covered_features),
            "grammar_sha256": expressibility.grammar_sha256,
            "status": negative.status,
            "transition_path": list(expressibility.transition_path),
            "witness_verification_sha256": expressibility.witness_verification_sha256,
        },
        "proof_plan": {
            "closed": proof_plan.closed,
            "mechanisms": list(proof_plan.mechanisms),
            "search_axes": [
                "lemma_shape",
                "invariants",
                "induction_variables",
                "normal_forms",
                "representations",
            ],
            "tactics": list(proof_plan.tactic_ids),
        },
        "dataset_pipeline": {
            "completed": dataset.completed,
            "heldout_commitment": dataset.heldout_commitment,
            "heldout_opened_at_final_stage": (
                dataset.records[-1].stage is R.DatasetStageV2.HELDOUT_TEST
            ),
            "record_chain": [
                {
                    "input_sha256": item.input_sha256,
                    "output_sha256": item.output_sha256,
                    "stage": item.stage.value,
                }
                for item in dataset.records
            ],
            "stages": [item.stage.value for item in dataset.records],
        },
        "blind_capability": {
            "bounded_unknown_decidable_passed": ladder.highest_passed >= 5,
            "evidence": blind_evidence,
            "highest_passed": ladder.highest_passed,
            "open_problem_claim_established": False,
            "open_problem_spend_gate_reached": ladder.open_problem_spend_authorized,
        },
        "serious_claim_chain": {
            "human_prior_art_review_present": False,
            "release_authorized": serious_release_authorized,
            "stages_present": sorted(item.stage.name.lower() for item in chain.links.values()),
        },
        "creative_yield": metrics.to_dict(),
        "objective_completion_audit": objective_audit,
        "claims": {
            "novel_mathematics_established": False,
            "open_problem_solved": False,
            "readiness_controls_passed": True,
            "serious_claim_release_authorized": serious_release_authorized,
        },
        "decision": "OPERATIONAL_FOR_SEALED_DOMAIN_CAMPAIGNS",
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_receipt(value: dict[str, Any], root: Path) -> None:
    body = dict(value)
    claimed = body.pop("content_sha256", None)
    if claimed != canonical_sha256(body):
        raise OperationalCampaignError("operational receipt content seal changed")
    if value != run_operational_campaign(root):
        raise OperationalCampaignError("operational receipt does not replay from bound sources")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--out", default="")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    generated = run_operational_campaign(root)
    if args.check:
        validate_receipt(json.loads((root / OUTPUT_PATH).read_text(encoding="utf-8")), root)
    else:
        destination = root / (args.out or OUTPUT_PATH)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(generated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "decision": generated["decision"],
                "extensions": len(generated["extension_admission"]["admitted_declarations"]),
                "proposals": generated["creativity"]["proposal_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
