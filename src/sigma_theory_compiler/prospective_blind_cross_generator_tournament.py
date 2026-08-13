"""Prospective three-world blind tournament across all native generator families."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

from .bayesian_candidate_generator import (
    BayesianBudget,
    BayesianCandidateGenerator,
    BayesianProposalBatch,
    BayesianState,
    ExactProbability,
    WeightedCandidate,
)
from .candidate_evaluation_ladder import (
    EvaluationLadder,
    EvaluationPhase,
    EvaluationStep,
    evaluate_candidate,
    validate_evaluation_replay,
)
from .candidate_generator_portfolio import build_generator_portfolio
from .candidate_knowledge_graph import (
    CandidateKnowledgeEdge,
    CandidateKnowledgeGraph,
    CandidateKnowledgeNode,
    KnowledgeEdgeKind,
    KnowledgeGraphLimits,
    KnowledgeNodeKind,
)
from .candidate_pareto_explanations import (
    MetricReceipt,
    ParetoLimits,
    build_pareto_explanations,
    validate_pareto_replay,
)
from .cross_domain_candidate_generator import (
    TransferLimits,
    generate_cross_domain_candidates,
    validate_transfer_replay,
)
from .egraph_candidate_generator import (
    SaturationLimits,
    extract_candidate_artifacts,
    saturate_expressions,
)
from .egraph_candidate_generator import validate_replay as validate_egraph_replay
from .evolutionary_candidate_generator import (
    EvaluationOutcome,
    EvolutionBudget,
    EvolutionRun,
    SeedStream,
    evolve_candidates,
)
from .grammar_candidate_generator import (
    GrammarLimits,
    GrammarSpec,
    generate_grammar_candidates,
    grammar_source_bindings,
    validate_grammar_manifest,
)
from .llm_candidate_generator import (
    RESPONSE_SCHEMA_VERSION,
    LLMBudgetState,
    LLMPolicy,
    LLMProposalManifest,
    LLMProposalRequest,
    generate_llm_candidates,
    llm_source_bindings,
    validate_llm_manifest,
)
from .math_expression_ir import add, literal, symbol
from .math_types import RATIONAL
from .sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    CheckResult,
    DomainPackDescriptor,
    DomainPackRef,
    GateDefinition,
    GateOutcome,
    OutcomeStatus,
    ProvenanceRecord,
    StageDefinition,
    StageOutcome,
    canonical_json_bytes,
    canonical_sha256,
)
from .symbolic_candidate_generator import (
    SymbolicCandidateGenerator,
    SymbolicGenerationBatch,
    SymbolicGeneratorBudget,
    SymbolicTemplate,
    SymbolicVariable,
)

CONFIG_SCHEMA = "sigma-prospective-blind-cross-generator-config-1.0"
RESULT_SCHEMA = "sigma-prospective-blind-cross-generator-result-1.0"
CAMPAIGN_ID = "prospective-blind-cross-generator-tournament-001"
CONFIG_PATH = "configs/prospective_blind_cross_generator_tournament.json"
SOURCE_PATH = "src/sigma_theory_compiler/prospective_blind_cross_generator_tournament.py"
TEST_PATH = "tests/test_prospective_blind_cross_generator_tournament.py"
OUTPUT_PATH = "runs/math/prospective-blind-cross-generator-tournament/campaign.json"
FAMILIES = ("bayesian", "cross_domain", "egraph", "evolutionary", "grammar", "llm", "symbolic")
KINDS = tuple(sorted(ArtifactKind, key=lambda item: item.value))
WORLD_ROWS = (
    (
        "prospective.modular_affine",
        "prospective-world-arc-20260813",
        "6dc10af7a574e74b07051e2c87975fae9abcba277fb7825aa620dd74c2358709",
    ),
    (
        "prospective.finite_difference",
        "prospective-world-lattice-20260813",
        "a5957f0ff438192b6a29c3201e1888472d4f21d1f57b1a83016119e33dfe6c70",
    ),
    (
        "prospective.graph_parity",
        "prospective-world-weave-20260813",
        "a2464946dbc725622418a0871e03b4acb88da9737c3f1f4e70a9250c9f367ba7",
    ),
)
_SEALED_TARGETS = {
    "prospective.modular_affine": 7,
    "prospective.finite_difference": 1,
    "prospective.graph_parity": 6,
}
CLAIMS = {
    "all_seven_native_generator_families_exercised_per_world": True,
    "all_generation_completed_before_target_unseal": True,
    "exactly_one_atomic_target_unseal_batch": True,
    "exactly_one_target_unseal_per_world": True,
    "holdout_cuts_validated": True,
    "only_all_hard_gate_pass_candidates_received_metrics": True,
    "only_all_hard_gate_pass_candidates_entered_pareto_fronts": True,
    "post_unseal_tuning_performed": False,
    "generator_output_establishes_truth": False,
    "corpus_absence_establishes_novelty": False,
    "pareto_rank_establishes_truth": False,
    "promotion_authorized": False,
}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("path is not a portable relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("path escapes project root") from error
    return resolved


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = canonical_sha256(body)
    return body


def _expected_config() -> dict[str, Any]:
    return {
        "budgets": {
            "candidates_per_family_per_world": 1,
            "generator_work_items_per_world": 128,
            "hypothesis_inventory_size": 11,
            "llm_calls_per_world": 1,
            "llm_maximum_micro_usd_per_world": 1_000,
            "maximum_pareto_work_units_per_world": 512,
        },
        "campaign_id": CAMPAIGN_ID,
        "common_domain_contract": {
            "hard_gates": ["hard_exact", "hard_holdout"],
            "stages": ["typed", "generated"],
        },
        "generator_families": list(FAMILIES),
        "policies": {
            "generator_target_access": "forbidden",
            "heuristic_truth_inference": "forbidden",
            "live_sqlite_access": "forbidden",
            "metrics_before_all_hard_gates_pass": "forbidden",
            "network_access": "forbidden",
            "post_unseal_tuning": "forbidden",
            "target_records_per_unseal": 3,
            "target_unseal_batches": 1,
        },
        "schema_version": CONFIG_SCHEMA,
        "worlds": [
            {"world_id": world_id, "public_seed": seed, "sealed_target_sha256": commitment}
            for world_id, seed, commitment in WORLD_ROWS
        ],
    }


def _load_config(root: Path, config_path: Path | None) -> dict[str, Any]:
    path = config_path or _resolve(root, CONFIG_PATH)
    if path.resolve() != _resolve(root, CONFIG_PATH):
        raise ValueError("tournament config path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != _expected_config():
        raise ValueError("tournament preregistration changed")
    portfolio = build_generator_portfolio()
    implemented = tuple(
        row["strategy_id"]
        for row in portfolio["capabilities"]
        if row["implementation_status"] == "implemented"
    )
    if implemented != FAMILIES:
        raise ValueError("native generator portfolio changed")
    return value


def _descriptor(world: Mapping[str, Any]) -> DomainPackDescriptor:
    return DomainPackDescriptor(
        world["world_id"],
        "1.0.0",
        KINDS,
        (
            StageDefinition("typed", 0, KINDS),
            StageDefinition("generated", 1, KINDS, ("typed",)),
        ),
        (
            GateDefinition("hard_exact", None, "typed", KINDS, ("typed",)),
            GateDefinition("hard_holdout", "typed", "generated", KINDS, ("generated", "typed")),
        ),
    )


def _public_world(world: Mapping[str, Any], inventory_size: int) -> dict[str, Any]:
    return {
        "world_id": world["world_id"],
        "public_seed": world["public_seed"],
        "hypothesis_inventory": list(range(inventory_size)),
        "target_disclosed": False,
    }


def _base_candidate(
    domain: DomainPackRef,
    world_id: str,
    family: str,
    variant: int,
    *,
    parents: Sequence[CandidateArtifact] = (),
) -> CandidateArtifact:
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        f"Target-blind {family} native proposal {variant} for {world_id}.",
        {"family": family, "variant": variant, "expression": ["x", "+", variant]},
        ProvenanceRecord.create(
            domain,
            {"campaign": CAMPAIGN_ID, "family": family, "variant": variant, "world": world_id},
            inputs=tuple(parent.ref for parent in parents),
        ),
        assumptions=("target unavailable during native generation",),
        claims=("requires_common_holdout_evaluation",),
    )


def _native_grammar(root: Path, domain: DomainPackRef) -> tuple[CandidateArtifact, dict[str, Any]]:
    manifest = generate_grammar_candidates(
        GrammarSpec(("x",), (), (), 1),
        GrammarLimits(4, 8, 8),
        domain,
        grammar_source_bindings(root),
    )
    validate_grammar_manifest(manifest.to_dict(), project_root=root)
    return manifest.candidates[0], manifest.to_dict()


def _native_symbolic(domain: DomainPackRef) -> tuple[CandidateArtifact, dict[str, Any]]:
    x, coefficient = sp.Symbol("x"), sp.Symbol("a")
    template = SymbolicTemplate.create(
        "prospective_linear",
        coefficient * x,
        variables=(SymbolicVariable("x", RATIONAL),),
        coefficient_symbols=("a",),
    )
    batch = SymbolicCandidateGenerator.generate(
        (template,),
        (1,),
        domain_pack=domain,
        budget=SymbolicGeneratorBudget(1, 1, 2, 2),
    )
    SymbolicGenerationBatch.from_dict(batch.to_dict())
    return batch.candidates[0], batch.to_dict()


def _native_evolutionary(
    domain: DomainPackRef, world_id: str, seed: str
) -> tuple[CandidateArtifact, dict[str, Any]]:
    initial = (
        _base_candidate(domain, world_id, "evolutionary", 0),
        _base_candidate(domain, world_id, "evolutionary", 1),
    )

    def mutate(parent: CandidateArtifact, stream: SeedStream) -> CandidateArtifact:
        variant = (parent.representation["variant"] + 1 + stream.draw(3)) % 11
        return _base_candidate(domain, world_id, "evolutionary", variant, parents=(parent,))

    def crossover(
        left: CandidateArtifact, right: CandidateArtifact, stream: SeedStream
    ) -> CandidateArtifact:
        variant = (
            left.representation["variant"] + right.representation["variant"] + stream.draw(2)
        ) % 11
        return _base_candidate(domain, world_id, "evolutionary", variant, parents=(left, right))

    def evaluate(artifact: CandidateArtifact) -> EvaluationOutcome:
        return EvaluationOutcome.create(
            artifact, OutcomeStatus.PASS, score=11 - artifact.representation["variant"]
        )

    run = evolve_candidates(
        initial,
        seed=seed,
        budget=EvolutionBudget(2, 1, 2, 4),
        mutate=mutate,
        crossover=crossover,
        evaluate=evaluate,
    )
    EvolutionRun.from_dict(run.to_dict())
    selected = max(run.artifacts, key=lambda item: (len(item.provenance.inputs), item.artifact_id))
    return selected, run.to_dict()


def _native_bayesian(
    domain: DomainPackRef, world_id: str, seed: str
) -> tuple[CandidateArtifact, dict[str, Any]]:
    seeds = (
        _base_candidate(domain, world_id, "bayesian", 0),
        _base_candidate(domain, world_id, "bayesian", 1),
    )
    budget = BayesianBudget(2, 1, 8)
    state = BayesianState.create(
        (
            WeightedCandidate(seeds[0], ExactProbability(1, 3)),
            WeightedCandidate(seeds[1], ExactProbability(2, 3)),
        ),
        budget,
    )
    numeric_seed = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    proposals = BayesianCandidateGenerator.propose(state, seed=numeric_seed, draws=8)
    BayesianProposalBatch.from_dict(proposals.to_dict())
    return proposals.proposals[0], {"state": state.to_dict(), "proposals": proposals.to_dict()}


def _native_egraph(
    domain: DomainPackRef, world_id: str
) -> tuple[CandidateArtifact, dict[str, Any]]:
    seed_expression = add(symbol("x"), literal(0))
    limits = SaturationLimits(64, 4, 2_000)
    result = saturate_expressions((seed_expression,), limits=limits)
    validate_egraph_replay(result, (seed_expression,), limits=limits)
    provenance = ProvenanceRecord.create(
        domain,
        {"campaign": CAMPAIGN_ID, "family": "egraph", "world": world_id},
    )
    return extract_candidate_artifacts(result, provenance)[0], result


def _native_cross_domain(
    domain: DomainPackRef, world_id: str
) -> tuple[CandidateArtifact, dict[str, Any]]:
    parents = tuple(
        CandidateArtifact.create(
            ArtifactKind.FORMULA,
            f"Prospective transfer source {index} for {world_id}.",
            {"expression": f"x+{index}", "variables": ["x"]},
            ProvenanceRecord.create(
                DomainPackRef(f"prospective.source_{label}", "1.0", digest * 64),
                {"campaign": CAMPAIGN_ID, "source": label, "world": world_id},
            ),
        )
        for index, label, digest in ((0, "a", "a"), (1, "b", "b"))
    )
    templates = ("formula_record_bundle_v1",)
    limits = TransferLimits(4, 2, 2, 2)
    result = generate_cross_domain_candidates(
        parents, domain, template_ids=templates, limits=limits
    )
    validate_transfer_replay(result, parents, domain, template_ids=templates, limits=limits)
    return CandidateArtifact.from_dict(result["candidates"][0]), result


def _native_llm(
    root: Path, domain: DomainPackRef, world_id: str, seed: str
) -> tuple[CandidateArtifact, dict[str, Any]]:
    policy = LLMPolicy(
        provider_id="offline.prospective.fixture",
        credential_env_var="PROSPECTIVE_TOURNAMENT_UNUSED_KEY",
        maximum_total_micro_usd=1_000,
        maximum_call_micro_usd=1_000,
        maximum_calls=1,
        maximum_prompt_tokens=32,
        maximum_completion_tokens=32,
        maximum_response_bytes=4_096,
        maximum_proposals=1,
    )
    request = LLMProposalRequest(
        request_id=f"prospective.{hashlib.sha256(world_id.encode()).hexdigest()[:12]}",
        prompt="Propose one bounded target-blind hypothesis token from public metadata.",
        prompt_token_count=9,
        completion_token_limit=16,
        deterministic_seed=int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16),
    )

    def provider(_request: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_id": request.request_id,
            "usage": {"prompt_tokens": 9, "completion_tokens": 7, "billed_micro_usd": 0},
            "proposals": [
                {
                    "proposal_id": f"proposal.{hashlib.sha256(seed.encode()).hexdigest()[:12]}",
                    "kind": "formula",
                    "statement": f"Offline target-blind proposal for {world_id}.",
                    "representation": {"dsl": "x+0", "quarantine": "required"},
                    "assumptions": ["target not present in provider request"],
                }
            ],
        }

    manifest = generate_llm_candidates(
        policy,
        request,
        LLMBudgetState(0, 0, 0, 0),
        domain,
        llm_source_bindings(root),
        provider,
    )
    validate_llm_manifest(manifest.to_dict(), request=request, project_root=root)
    LLMProposalManifest.from_dict(manifest.to_dict())
    return manifest.candidates[0], manifest.to_dict()


def _generate_native(
    root: Path, world: Mapping[str, Any], descriptor: DomainPackDescriptor
) -> tuple[dict[str, CandidateArtifact], dict[str, Any]]:
    domain = descriptor.ref
    world_id, seed = world["world_id"], world["public_seed"]
    generated = {
        "bayesian": _native_bayesian(domain, world_id, seed),
        "cross_domain": _native_cross_domain(domain, world_id),
        "egraph": _native_egraph(domain, world_id),
        "evolutionary": _native_evolutionary(domain, world_id, seed),
        "grammar": _native_grammar(root, domain),
        "llm": _native_llm(root, domain, world_id, seed),
        "symbolic": _native_symbolic(domain),
    }
    candidates = {family: row[0] for family, row in generated.items()}
    if (
        tuple(sorted(candidates)) != FAMILIES
        or len({row.artifact_id for row in candidates.values()}) != 7
    ):
        raise ValueError("native generator coverage changed")
    if any(row.provenance.domain_pack != domain for row in candidates.values()):
        raise ValueError("native candidate escaped prospective world domain")
    receipts = {
        family: {
            "native_candidate": candidate.ref.to_dict(),
            "receipt_sha256": canonical_sha256(generated[family][1]),
        }
        for family, candidate in candidates.items()
    }
    return candidates, receipts


def _adapt_candidates(
    native: Mapping[str, CandidateArtifact],
    descriptor: DomainPackDescriptor,
    public: Mapping[str, Any],
    inventory_size: int,
) -> dict[str, CandidateArtifact]:
    public_sha = canonical_sha256(public)
    adapted = {}
    for family in FAMILIES:
        parent = native[family]
        hypothesis = (
            int(
                canonical_sha256(
                    {
                        "adapter": "target-blind-content-map-v1",
                        "native": parent.ref.to_dict(),
                        "public": public_sha,
                    }
                )[:16],
                16,
            )
            % inventory_size
        )
        adapted[family] = CandidateArtifact.create(
            ArtifactKind.CONJECTURE,
            f"{family} proposes registered hypothesis {hypothesis} for {public['world_id']}.",
            {
                "adapter": "target-blind-content-map-v1",
                "family": family,
                "hypothesis": hypothesis,
                "inventory_size": inventory_size,
                "native_candidate": parent.ref.to_dict(),
                "public_world_sha256": public_sha,
                "target_fields_read": [],
            },
            ProvenanceRecord.create(
                descriptor.ref,
                {
                    "adapter": "target-blind-content-map-v1",
                    "campaign": CAMPAIGN_ID,
                    "family": family,
                    "world": public["world_id"],
                },
                inputs=(parent.ref,),
            ),
            assumptions=("closed public hypothesis inventory only",),
            claims=("requires_single_unseal_holdout_evaluation",),
        )
    return adapted


def _unseal_targets(
    worlds: Sequence[Mapping[str, Any]], accesses: Counter[str]
) -> dict[str, dict[str, Any]]:
    if tuple(world["world_id"] for world in worlds) != tuple(row[0] for row in WORLD_ROWS):
        raise ValueError("atomic target-unseal inventory changed")
    targets = {}
    for world in worlds:
        world_id = world["world_id"]
        accesses[world_id] += 1
        target = {"hypothesis": _SEALED_TARGETS[world_id], "world_id": world_id}
        if canonical_sha256(target) != world["sealed_target_sha256"]:
            raise ValueError("sealed target commitment mismatch")
        targets[world_id] = target
    return targets


def _holdout(world: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    axiom = CandidateKnowledgeNode.create(
        KnowledgeNodeKind.AXIOM,
        {"public_seed_sha256": hashlib.sha256(world["public_seed"].encode()).hexdigest()},
    )
    theorem = CandidateKnowledgeNode.create(KnowledgeNodeKind.THEOREM, dict(target))
    proof = CandidateKnowledgeNode.create(
        KnowledgeNodeKind.PROOF,
        {"role": "sealed_reference_witness", "target_sha256": canonical_sha256(target)},
    )
    graph = CandidateKnowledgeGraph.create(
        f"prospective.{world['world_id']}.sealed",
        artifacts=(),
        nodes=(axiom, theorem, proof),
        edges=(
            CandidateKnowledgeEdge.create(
                KnowledgeEdgeKind.DEPENDENCY, theorem.node_id, axiom.node_id
            ),
            CandidateKnowledgeEdge.create(KnowledgeEdgeKind.PROVES, proof.node_id, theorem.node_id),
        ),
        limits=KnowledgeGraphLimits(1, 8, 8, 4, 4),
    )
    cut = graph.holdout_cut(theorem.node_id)
    cut.validate_against(graph)
    if set(cut.visible_node_ids) != {axiom.node_id} or set(cut.forbidden_node_ids) != {
        theorem.node_id,
        proof.node_id,
    }:
        raise ValueError("prospective holdout cut changed")
    return {"graph": graph.to_dict(), "cut": cut.to_dict()}


class _TournamentPack:
    def __init__(self, descriptor: DomainPackDescriptor, target: int) -> None:
        self._descriptor = descriptor
        self._target = target

    @property
    def descriptor(self) -> DomainPackDescriptor:
        return self._descriptor

    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: Mapping[str, StageOutcome],
    ) -> StageOutcome:
        representation = artifact.representation
        valid = (
            set(representation)
            == {
                "adapter",
                "family",
                "hypothesis",
                "inventory_size",
                "native_candidate",
                "public_world_sha256",
                "target_fields_read",
            }
            and representation["target_fields_read"] == []
            and isinstance(representation["hypothesis"], int)
            and 0 <= representation["hypothesis"] < representation["inventory_size"] == 11
        )
        check = CheckResult.create(
            f"{stage.stage_id}.closed_target_blind_candidate",
            valid,
            {"artifact": artifact.artifact_id, "prior": sorted(prior_outcomes)},
        )
        status = OutcomeStatus.PASS if valid else OutcomeStatus.REJECT
        return StageOutcome.create(
            stage.stage_id,
            artifact.ref,
            status,
            (check,),
            reason_codes=() if valid else ("invalid_target_blind_candidate",),
        )

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> GateOutcome:
        passed = (
            gate.gate_id == "hard_exact" or artifact.representation["hypothesis"] == self._target
        )
        check = CheckResult.create(
            f"{gate.gate_id}.prospective_contract",
            passed,
            {
                "artifact": artifact.artifact_id,
                "candidate_hypothesis": artifact.representation["hypothesis"],
                "target_compared": gate.gate_id == "hard_holdout",
            },
        )
        status = OutcomeStatus.PASS if passed else OutcomeStatus.REJECT
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            status,
            tuple(stage_outcomes[item].ref for item in sorted(stage_outcomes)),
            (check,),
            reason_codes=() if passed else ("sealed_holdout_counterexample",),
        )


def _ladder(descriptor: DomainPackDescriptor) -> EvaluationLadder:
    return EvaluationLadder.create(
        descriptor,
        (
            EvaluationStep("typed", "hard_exact", EvaluationPhase.CHEAP),
            EvaluationStep("generated", "hard_holdout", EvaluationPhase.FORMAL),
        ),
    )


def _metrics(candidates: Sequence[CandidateArtifact]) -> list[MetricReceipt]:
    rows = []
    for candidate in candidates:
        values = {
            "lineage_inputs": len(candidate.provenance.inputs),
            "representation_bytes": len(canonical_json_bytes(candidate.representation)),
        }
        for metric_id, direction in (
            ("lineage_inputs", "maximize"),
            ("representation_bytes", "minimize"),
        ):
            rows.append(
                MetricReceipt.create(
                    candidate.ref,
                    metric_id,
                    direction,
                    values[metric_id],
                    canonical_sha256(
                        {
                            "candidate": candidate.ref.to_dict(),
                            "metric": metric_id,
                            "value": values[metric_id],
                        }
                    ),
                )
            )
    return rows


def _world_result(
    world: Mapping[str, Any],
    descriptor: DomainPackDescriptor,
    candidates: Mapping[str, CandidateArtifact],
    native_receipts: Mapping[str, Any],
    target: Mapping[str, Any],
    holdout: Mapping[str, Any],
    maximum_pareto_work: int,
) -> dict[str, Any]:
    pack = _TournamentPack(descriptor, target["hypothesis"])
    ladder = _ladder(descriptor)
    evaluations = {}
    eligible = []
    eligible_gates = []
    for family in FAMILIES:
        candidate = candidates[family]
        evaluation = evaluate_candidate(pack, candidate, ladder)
        validate_evaluation_replay(evaluation, pack, candidate)
        evaluations[family] = evaluation
        if evaluation["all_required_gates_passed"]:
            eligible.append(candidate)
            eligible_gates.extend(GateOutcome.from_dict(row) for row in evaluation["gate_outcomes"])
    metrics = _metrics(eligible)
    pareto = None
    if eligible:
        limits = ParetoLimits(7, 2, 2, maximum_pareto_work)
        directions = {"lineage_inputs": "maximize", "representation_bytes": "minimize"}
        pareto = build_pareto_explanations(
            eligible,
            eligible_gates,
            metrics,
            required_gate_ids=("hard_exact", "hard_holdout"),
            metric_directions=directions,
            limits=limits,
        )
        validate_pareto_replay(
            pareto,
            eligible,
            eligible_gates,
            metrics,
            required_gate_ids=("hard_exact", "hard_holdout"),
            metric_directions=directions,
            limits=limits,
        )
    statuses = Counter(row["status"] for row in evaluations.values())
    return {
        "world_id": world["world_id"],
        "public_seed": world["public_seed"],
        "sealed_target_sha256": world["sealed_target_sha256"],
        "unsealed_target": dict(target),
        "domain_pack": descriptor.to_dict(),
        "holdout": dict(holdout),
        "native_generator_receipts": dict(native_receipts),
        "candidates": [candidates[family].to_dict() for family in FAMILIES],
        "family_bindings": [
            {"family": family, "candidate": candidates[family].ref.to_dict()} for family in FAMILIES
        ],
        "evaluations": evaluations,
        "terminal_status_counts": dict(sorted(statuses.items())),
        "pareto_eligible_families": [
            family for family in FAMILIES if evaluations[family]["all_required_gates_passed"]
        ],
        "metric_receipts": [row.to_dict() for row in metrics],
        "pareto": pareto,
        "decision": (
            "pass_at_least_one_target_blind_candidate_survived"
            if eligible
            else "reject_fixed_budget_exhausted_without_holdout_match"
        ),
    }


def build_campaign(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root, config_path)
    inventory_size = config["budgets"]["hypothesis_inventory_size"]
    prepared = []
    phase_events = []
    for world in config["worlds"]:
        descriptor = _descriptor(world)
        public = _public_world(world, inventory_size)
        native, native_receipts = _generate_native(root, world, descriptor)
        candidates = _adapt_candidates(native, descriptor, public, inventory_size)
        phase_events.extend(
            {"event": "generated", "family": family, "world_id": world["world_id"]}
            for family in FAMILIES
        )
        prepared.append((world, descriptor, candidates, native_receipts))

    accesses: Counter[str] = Counter()
    targets = _unseal_targets(config["worlds"], accesses)
    phase_events.append(
        {
            "event": "targets_unsealed",
            "world_ids": [world["world_id"] for world in config["worlds"]],
        }
    )
    results = []
    for world, descriptor, candidates, native_receipts in prepared:
        target = targets[world["world_id"]]
        holdout = _holdout(world, target)
        results.append(
            _world_result(
                world,
                descriptor,
                candidates,
                native_receipts,
                target,
                holdout,
                config["budgets"]["maximum_pareto_work_units_per_world"],
            )
        )

    if any(accesses[world_id] != 1 for world_id, _, _ in WORLD_ROWS):
        raise ValueError("target unseal count changed")
    first_unseal = next(
        index for index, row in enumerate(phase_events) if row["event"] == "targets_unsealed"
    )
    if first_unseal != 21 or any(
        row["event"] == "generated" for row in phase_events[first_unseal:]
    ):
        raise ValueError("generation did not precede the single unseal phase")
    status_counts = Counter(result["decision"].split("_", 1)[0] for result in results)
    eligible_count = sum(len(result["pareto_eligible_families"]) for result in results)
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "completed_preregistered_three_world_single_unseal_tournament",
        "preregistration": config,
        "portfolio": build_generator_portfolio(),
        "world_results": results,
        "phase_ledger": {
            "events": phase_events,
            "generation_events_before_first_unseal": first_unseal,
            "pre_unseal_target_access_count": 0,
            "post_unseal_generation_count": 0,
            "target_records_unsealed": dict(sorted(accesses.items())),
            "target_unseal_batches": 1,
            "post_unseal_tuning_events": 0,
        },
        "counts": {
            "worlds": 3,
            "generator_families": 7,
            "selected_candidates": 21,
            "evaluation_results": 21,
            "target_records_unsealed": 3,
            "target_unseal_batches": 1,
            "pareto_eligible_candidates": eligible_count,
            "world_passes": status_counts["pass"],
            "world_rejects": status_counts["reject"],
            "world_blocks": status_counts["block"],
        },
        "claims": dict(CLAIMS),
        "scope": (
            "exactly three preregistered deterministic finite worlds and fixed native-generator "
            "budgets; target-blind content mapping is a bounded search adapter, and tournament "
            "success does not establish general discovery, novelty, truth, or promotion"
        ),
        "next_gate": (
            "repeat_on_independently_authored_external_worlds_with_nontrivial_kernel_proofs_"
            "without_changing_this_tournament_after_unseal"
        ),
        "source_bindings": {
            label: {"path": path, "file_sha256": _file_sha(_resolve(root, path))}
            for label, path in (
                ("config", CONFIG_PATH),
                ("source", SOURCE_PATH),
                ("test", TEST_PATH),
            )
        },
    }
    return _seal(body)


def validate_campaign(
    value: Mapping[str, Any], root: Path, config_path: Path | None = None
) -> None:
    if value.get("schema_version") != RESULT_SCHEMA or value.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("prospective tournament identity changed")
    if value.get("content_sha256") != canonical_sha256(
        {key: child for key, child in value.items() if key != "content_sha256"}
    ):
        raise ValueError("prospective tournament self-seal changed")
    if value.get("claims") != CLAIMS:
        raise ValueError("prospective tournament claim boundary changed")
    if dict(value) != build_campaign(root, config_path):
        raise ValueError("prospective tournament immutable replay mismatch")


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    result = build_campaign(root, config_path)
    validate_campaign(result, root, config_path)
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable prospective tournament artifact differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    result = run(root, _resolve(root, arguments.config))
    _write_immutable(_resolve(root, arguments.output), result)
    print(json.dumps({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
