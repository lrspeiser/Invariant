"""Cross-generator Sigma Core hard-gate, Pareto, explanation, and replay campaign."""

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
from .candidate_generator_portfolio import build_generator_portfolio
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
from .egraph_candidate_generator import (
    validate_replay as validate_egraph_replay,
)
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
    run_gate,
    run_stage,
)
from .symbolic_candidate_generator import (
    SymbolicCandidateGenerator,
    SymbolicGenerationBatch,
    SymbolicGeneratorBudget,
    SymbolicTemplate,
    SymbolicVariable,
)

CONFIG_SCHEMA = "sigma-comprehensive-alpha-cross-generator-config-1.0"
RESULT_SCHEMA = "sigma-comprehensive-alpha-cross-generator-result-1.0"
CAMPAIGN_ID = "comprehensive-alpha-cross-generator-001"
CONFIG_PATH = "configs/comprehensive_alpha_cross_generator_campaign.json"
SOURCE_PATH = "src/sigma_theory_compiler/comprehensive_alpha_cross_generator_campaign.py"
TEST_PATH = "tests/test_comprehensive_alpha_cross_generator_campaign.py"
OUTPUT_PATH = "runs/math/comprehensive-alpha-cross-generator/campaign.json"
FAMILIES = ("bayesian", "cross_domain", "egraph", "evolutionary", "grammar", "llm", "symbolic")
KINDS = tuple(sorted(ArtifactKind, key=lambda item: item.value))
DESCRIPTOR = DomainPackDescriptor(
    "synthetic.comprehensive_alpha",
    "1.0.0",
    KINDS,
    (
        StageDefinition("typed", 0, KINDS),
        StageDefinition("exact", 1, KINDS, ("typed",)),
    ),
    (
        GateDefinition("hard_exact", "typed", "exact", KINDS, ("exact", "typed")),
        GateDefinition("hard_structure", "typed", "exact", KINDS, ("exact", "typed")),
    ),
)
CLAIMS = {
    "all_implemented_generator_families_exercised": True,
    "common_sigma_core_domain_pack_used": True,
    "all_candidates_received_complete_hard_gate_coverage": True,
    "only_all_hard_gate_pass_candidates_received_metrics": True,
    "only_all_hard_gate_pass_candidates_entered_pareto_fronts": True,
    "pass_block_reject_error_preserved": True,
    "deterministic_replay_validated": True,
    "generator_output_establishes_truth": False,
    "heuristic_score_establishes_truth": False,
    "pareto_rank_establishes_truth": False,
    "novelty_established": False,
    "promotion_authorized": False,
    "external_benchmark_success_established": False,
}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("path is not a nonempty portable relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("path escapes project root") from error
    return path


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = canonical_sha256(body)
    return body


def _expected_config() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generator_families": list(FAMILIES),
        "common_domain_pack": {
            "pack_id": "synthetic.comprehensive_alpha",
            "pack_version": "1.0.0",
            "stages": ["typed", "exact"],
            "hard_gates": ["hard_exact", "hard_structure"],
        },
        "preregistered_gate_outcomes": {
            "bayesian": {"hard_exact": "pass", "hard_structure": "pass"},
            "cross_domain": {"hard_exact": "pass", "hard_structure": "block"},
            "egraph": {"hard_exact": "reject", "hard_structure": "pass"},
            "evolutionary": {"hard_exact": "pass", "hard_structure": "pass"},
            "grammar": {"hard_exact": "pass", "hard_structure": "pass"},
            "llm": {"hard_exact": "error", "hard_structure": "pass"},
            "symbolic": {"hard_exact": "pass", "hard_structure": "pass"},
        },
        "exact_soft_metrics": {
            "provenance_inputs": "maximize",
            "representation_bytes": "minimize",
        },
        "budgets": {
            "maximum_candidates": 7,
            "maximum_generator_work_items": 128,
            "maximum_pareto_work_units": 128,
            "llm_calls": 1,
            "llm_maximum_micro_usd": 1000,
        },
        "policies": {
            "network_access": "forbidden",
            "live_sqlite_access": "forbidden",
            "llm_provider": "deterministic_offline_fixture",
            "metrics_before_all_hard_gates_pass": "forbidden",
            "generator_self_promotion": "forbidden",
            "heuristic_truth_inference": "forbidden",
        },
        "output_path": OUTPUT_PATH,
    }


def _load_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _resolve(root, CONFIG_PATH)
    if path.resolve() != _resolve(root, CONFIG_PATH):
        raise ValueError("comprehensive alpha config path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != _expected_config():
        raise ValueError("comprehensive alpha config contract changed")
    portfolio = build_generator_portfolio()
    implemented = tuple(
        row["strategy_id"]
        for row in portfolio["capabilities"]
        if row["implementation_status"] == "implemented"
    )
    if implemented != FAMILIES:
        raise ValueError("implemented generator portfolio changed")
    return value


def _candidate(
    family: str,
    variant: int,
    *,
    domain: DomainPackRef = DESCRIPTOR.ref,
    parents: Sequence[CandidateArtifact] = (),
) -> CandidateArtifact:
    provenance = ProvenanceRecord.create(
        domain,
        {"campaign": CAMPAIGN_ID, "family": family, "variant": variant},
        inputs=tuple(parent.ref for parent in parents),
    )
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        f"Bounded comprehensive-alpha {family} candidate {variant}.",
        {"family": family, "variant": variant, "expression": ["x", "+", variant]},
        provenance,
        assumptions=("syntactic bounded campaign candidate only",),
        claims=("requires_sigma_core_hard_gates",),
    )


def _generate_grammar(root: Path) -> tuple[CandidateArtifact, dict[str, Any]]:
    spec = GrammarSpec(("x",), (), (), 1)
    limits = GrammarLimits(4, 8, 8)
    manifest = generate_grammar_candidates(
        spec, limits, DESCRIPTOR.ref, grammar_source_bindings(root)
    )
    validate_grammar_manifest(manifest.to_dict(), project_root=root)
    return manifest.candidates[0], manifest.to_dict()


def _generate_symbolic() -> tuple[CandidateArtifact, dict[str, Any]]:
    x, a = sp.Symbol("x"), sp.Symbol("a")
    template = SymbolicTemplate.create(
        "alpha_linear",
        a * x,
        variables=(SymbolicVariable("x", RATIONAL),),
        coefficient_symbols=("a",),
    )
    batch = SymbolicCandidateGenerator.generate(
        (template,),
        (1,),
        domain_pack=DESCRIPTOR.ref,
        budget=SymbolicGeneratorBudget(1, 1, 2, 2),
    )
    SymbolicGenerationBatch.from_dict(batch.to_dict())
    return batch.candidates[0], batch.to_dict()


def _generate_evolutionary() -> tuple[CandidateArtifact, dict[str, Any]]:
    initial = (_candidate("evolutionary", 0), _candidate("evolutionary", 1))

    def mutate(parent: CandidateArtifact, stream: SeedStream) -> CandidateArtifact:
        variant = (parent.representation["variant"] + 1 + stream.draw(3)) % 7
        return _candidate("evolutionary", variant, parents=(parent,))

    def crossover(
        left: CandidateArtifact, right: CandidateArtifact, stream: SeedStream
    ) -> CandidateArtifact:
        variant = (
            left.representation["variant"] + right.representation["variant"] + stream.draw(2)
        ) % 7
        return _candidate("evolutionary", variant, parents=(left, right))

    def evaluate(artifact: CandidateArtifact) -> EvaluationOutcome:
        return EvaluationOutcome.create(
            artifact, OutcomeStatus.PASS, score=10 - artifact.representation["variant"]
        )

    run = evolve_candidates(
        initial,
        seed="comprehensive-alpha-evolution-001",
        budget=EvolutionBudget(2, 1, 2, 4),
        mutate=mutate,
        crossover=crossover,
        evaluate=evaluate,
    )
    EvolutionRun.from_dict(run.to_dict())
    selected = max(run.artifacts, key=lambda item: (len(item.provenance.inputs), item.artifact_id))
    return selected, run.to_dict()


def _generate_bayesian() -> tuple[CandidateArtifact, dict[str, Any]]:
    seeds = (_candidate("bayesian", 0), _candidate("bayesian", 1))
    budget = BayesianBudget(2, 1, 8)
    state = BayesianState.create(
        (
            WeightedCandidate(seeds[0], ExactProbability(1, 3)),
            WeightedCandidate(seeds[1], ExactProbability(2, 3)),
        ),
        budget,
    )
    proposals = BayesianCandidateGenerator.propose(state, seed=20260812, draws=8)
    BayesianProposalBatch.from_dict(proposals.to_dict())
    selected = proposals.proposals[0]
    return selected, {"state": state.to_dict(), "proposals": proposals.to_dict()}


def _generate_egraph() -> tuple[CandidateArtifact, dict[str, Any]]:
    seed = add(symbol("x"), literal(0))
    limits = SaturationLimits(64, 4, 2_000)
    result = saturate_expressions((seed,), limits=limits)
    validate_egraph_replay(result, (seed,), limits=limits)
    provenance = ProvenanceRecord.create(
        DESCRIPTOR.ref,
        {"campaign": CAMPAIGN_ID, "family": "egraph", "result": result["content_sha256"]},
    )
    candidates = extract_candidate_artifacts(result, provenance)
    return candidates[0], result


def _generate_cross_domain() -> tuple[CandidateArtifact, dict[str, Any]]:
    parents = tuple(
        CandidateArtifact.create(
            ArtifactKind.FORMULA,
            f"Cross-domain source formula {index}.",
            {"expression": f"x+{index}", "variables": ["x"]},
            ProvenanceRecord.create(
                DomainPackRef(f"synthetic.alpha_source_{label}", "1.0", digest * 64),
                {"campaign": CAMPAIGN_ID, "source": label},
            ),
        )
        for index, label, digest in ((0, "a", "a"), (1, "b", "b"))
    )
    templates = ("formula_record_bundle_v1",)
    limits = TransferLimits(4, 2, 2, 2)
    result = generate_cross_domain_candidates(
        parents, DESCRIPTOR.ref, template_ids=templates, limits=limits
    )
    validate_transfer_replay(result, parents, DESCRIPTOR.ref, template_ids=templates, limits=limits)
    return CandidateArtifact.from_dict(result["candidates"][0]), result


def _generate_llm(root: Path) -> tuple[CandidateArtifact, dict[str, Any]]:
    policy = LLMPolicy(
        provider_id="offline.fixture",
        credential_env_var="COMPREHENSIVE_ALPHA_UNUSED_KEY",
        maximum_total_micro_usd=1_000,
        maximum_call_micro_usd=1_000,
        maximum_calls=1,
        maximum_prompt_tokens=32,
        maximum_completion_tokens=32,
        maximum_response_bytes=4_096,
        maximum_proposals=1,
    )
    request = LLMProposalRequest(
        request_id="comprehensive.alpha.0001",
        prompt="Propose one bounded quarantined syntactic formula.",
        prompt_token_count=7,
        completion_token_limit=16,
        deterministic_seed=20260812,
    )
    budget = LLMBudgetState(0, 0, 0, 0)

    def provider(_request: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_id": request.request_id,
            "usage": {"prompt_tokens": 7, "completion_tokens": 8, "billed_micro_usd": 0},
            "proposals": [
                {
                    "proposal_id": "alpha.offline.001",
                    "kind": "formula",
                    "statement": "A quarantined offline comprehensive-alpha proposal.",
                    "representation": {"dsl": "x+x", "quarantine": "required"},
                    "assumptions": ["syntactic proposal only"],
                }
            ],
        }

    sources = llm_source_bindings(root)
    manifest = generate_llm_candidates(policy, request, budget, DESCRIPTOR.ref, sources, provider)
    validate_llm_manifest(manifest.to_dict(), request=request, project_root=root)
    LLMProposalManifest.from_dict(manifest.to_dict())
    return manifest.candidates[0], manifest.to_dict()


def _generate_all(root: Path) -> tuple[dict[str, CandidateArtifact], dict[str, Any]]:
    generated = {
        "bayesian": _generate_bayesian(),
        "cross_domain": _generate_cross_domain(),
        "egraph": _generate_egraph(),
        "evolutionary": _generate_evolutionary(),
        "grammar": _generate_grammar(root),
        "llm": _generate_llm(root),
        "symbolic": _generate_symbolic(),
    }
    candidates = {family: row[0] for family, row in generated.items()}
    if (
        tuple(sorted(candidates)) != FAMILIES
        or len({item.artifact_id for item in candidates.values()}) != 7
    ):
        raise ValueError("generator family candidate coverage changed")
    if any(item.provenance.domain_pack != DESCRIPTOR.ref for item in candidates.values()):
        raise ValueError("generator candidate escaped common domain pack")
    return candidates, {family: row[1] for family, row in generated.items()}


class _ComprehensiveAlphaPack:
    def __init__(self, classifications: Mapping[str, Mapping[str, str]]) -> None:
        self._classifications = classifications

    @property
    def descriptor(self) -> DomainPackDescriptor:
        return DESCRIPTOR

    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: Mapping[str, StageOutcome],
    ) -> StageOutcome:
        check = CheckResult.create(
            f"{stage.stage_id}.closed_candidate",
            True,
            {"artifact": artifact.artifact_id, "prior": sorted(prior_outcomes)},
        )
        return StageOutcome.create(stage.stage_id, artifact.ref, OutcomeStatus.PASS, (check,))

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> GateOutcome:
        status_name = self._classifications[artifact.artifact_id][gate.gate_id]
        if status_name == "error":
            raise RuntimeError("sealed domain-pack error fixture")
        status = OutcomeStatus(status_name)
        passed = status is OutcomeStatus.PASS
        check = CheckResult.create(
            f"{gate.gate_id}.campaign_contract",
            passed,
            {"artifact": artifact.artifact_id, "registered_status": status.value},
        )
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            status,
            tuple(stage_outcomes[item].ref for item in sorted(stage_outcomes)),
            (check,),
            reason_codes=() if passed else (f"registered_{status.value}_fixture",),
        )


def _evaluate(
    candidates: Mapping[str, CandidateArtifact], config: Mapping[str, Any]
) -> tuple[list[StageOutcome], list[GateOutcome]]:
    classifications = {
        candidates[family].artifact_id: config["preregistered_gate_outcomes"][family]
        for family in FAMILIES
    }
    pack = _ComprehensiveAlphaPack(classifications)
    stages: list[StageOutcome] = []
    gates: list[GateOutcome] = []
    for family in FAMILIES:
        candidate = candidates[family]
        typed = run_stage(pack, candidate, "typed")
        exact = run_stage(pack, candidate, "exact", {"typed": typed})
        stage_map = {"typed": typed, "exact": exact}
        stages.extend((typed, exact))
        gates.extend(
            run_gate(pack, candidate, gate_id, stage_map)
            for gate_id in ("hard_exact", "hard_structure")
        )
    return stages, gates


def _metric_receipts(candidates: Sequence[CandidateArtifact]) -> list[MetricReceipt]:
    receipts = []
    for candidate in candidates:
        values = {
            "provenance_inputs": len(candidate.provenance.inputs),
            "representation_bytes": len(canonical_json_bytes(candidate.representation)),
        }
        for metric_id, direction in (
            ("provenance_inputs", "maximize"),
            ("representation_bytes", "minimize"),
        ):
            receipts.append(
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
    return receipts


def build_campaign(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = _load_config(root, config_path)
    candidates, generator_receipts = _generate_all(root)
    stages, gates = _evaluate(candidates, config)
    gate_by_key = {(row.artifact.artifact_id, row.gate_id): row for row in gates}
    eligible_families = [
        family
        for family in FAMILIES
        if all(
            gate_by_key[candidates[family].artifact_id, gate_id].status is OutcomeStatus.PASS
            for gate_id in ("hard_exact", "hard_structure")
        )
    ]
    eligible = [candidates[family] for family in eligible_families]
    metrics = _metric_receipts(eligible)
    eligible_gates = [
        gate
        for gate in gates
        if gate.artifact.artifact_id in {candidate.artifact_id for candidate in eligible}
    ]
    limits = ParetoLimits(7, 2, 2, config["budgets"]["maximum_pareto_work_units"])
    pareto = build_pareto_explanations(
        eligible,
        eligible_gates,
        metrics,
        required_gate_ids=("hard_exact", "hard_structure"),
        metric_directions=config["exact_soft_metrics"],
        limits=limits,
    )
    validate_pareto_replay(
        pareto,
        eligible,
        eligible_gates,
        metrics,
        required_gate_ids=("hard_exact", "hard_structure"),
        metric_directions=config["exact_soft_metrics"],
        limits=limits,
    )
    family_by_id = {candidate.artifact_id: family for family, candidate in candidates.items()}
    excluded = []
    for family in FAMILIES:
        if family in eligible_families:
            continue
        family_gates = sorted(
            (row for row in gates if row.artifact == candidates[family].ref),
            key=lambda row: row.gate_id,
        )
        body = {
            "family": family,
            "candidate": candidates[family].ref.to_dict(),
            "hard_gate_statuses": {row.gate_id: row.status.value for row in family_gates},
            "metric_receipts": [],
            "pareto_front": None,
            "reason": "not_all_required_hard_gates_passed",
            "truth_established": False,
            "promotion_authorized": False,
        }
        excluded.append({**body, "explanation_sha256": canonical_sha256(body)})
    status_counts = Counter(row.status.value for row in gates)
    result = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "completed_cross_generator_hard_gate_pareto_replay_with_fail_closed_exclusions",
        "portfolio": build_generator_portfolio(),
        "domain_pack": DESCRIPTOR.to_dict(),
        "generator_receipts": generator_receipts,
        "selected_candidates": [candidates[family].to_dict() for family in FAMILIES],
        "candidate_family_bindings": [
            {"family": family, "candidate": candidates[family].ref.to_dict()} for family in FAMILIES
        ],
        "stage_outcomes": [row.to_dict() for row in stages],
        "gate_outcomes": [row.to_dict() for row in gates],
        "hard_gate_status_counts": dict(sorted(status_counts.items())),
        "pareto_eligible_families": eligible_families,
        "metric_receipts": [row.to_dict() for row in metrics],
        "pareto": pareto,
        "hard_gate_exclusion_explanations": excluded,
        "counts": {
            "generator_families": len(FAMILIES),
            "selected_candidates": len(candidates),
            "stage_outcomes": len(stages),
            "gate_outcomes": len(gates),
            "pareto_eligible_candidates": len(eligible),
            "hard_gate_excluded_candidates": len(candidates) - len(eligible),
            "metric_receipts": len(metrics),
        },
        "claims": dict(CLAIMS),
        "scope": (
            "one deterministic cross-generator integration campaign across the seven implemented "
            "generator families and one common Sigma Core domain pack; outcomes and syntactic "
            "metrics are integration evidence only, not truth, novelty, promotion, or external "
            "benchmark success"
        ),
        "first_remaining_blocker": (
            "run_the_same_cross_generator_contract_against_independently_registered_held_out_"
            "domain_packs_and_external_proof_oracles_without_leaking_targets"
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
    if set(family_by_id.values()) != set(FAMILIES):
        raise ValueError("candidate family binding changed")
    return _seal(result)


def validate_campaign(
    value: Mapping[str, Any], root: Path, config_path: Path | None = None
) -> None:
    if value.get("schema_version") != RESULT_SCHEMA or value.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("comprehensive alpha result identity changed")
    if value.get("content_sha256") != canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    ):
        raise ValueError("comprehensive alpha result self-seal changed")
    if value.get("claims") != CLAIMS or value.get("counts") != {
        "generator_families": 7,
        "selected_candidates": 7,
        "stage_outcomes": 14,
        "gate_outcomes": 14,
        "pareto_eligible_candidates": 4,
        "hard_gate_excluded_candidates": 3,
        "metric_receipts": 8,
    }:
        raise ValueError("comprehensive alpha result contract changed")
    if dict(value) != build_campaign(root, config_path):
        raise ValueError("comprehensive alpha immutable replay mismatch")


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    result = build_campaign(root, config_path)
    validate_campaign(result, root, config_path)
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable comprehensive alpha artifact differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()


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
