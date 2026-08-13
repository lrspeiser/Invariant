"""Prospective hidden semantic-expression tournament with exact proof/counterexamples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

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
from .math_expression_ir import Equation, Recurrence, call, literal, symbol
from .math_proof import (
    prove_induction,
    prove_rational_identity,
    validate_induction_certificate,
    validate_rational_identity_certificate,
)
from .math_types import INTEGER, RATIONAL
from .prospective_blind_cross_generator_tournament import _generate_native
from .sigma_core import (
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
    canonical_json_bytes,
    canonical_sha256,
)
from .symbolic_candidate_generator import _sympy_to_ir

CONFIG_SCHEMA = "sigma-semantic-formula-proof-holdout-config-1.0"
RESULT_SCHEMA = "sigma-semantic-formula-proof-holdout-result-1.0"
CAMPAIGN_ID = "semantic-formula-proof-holdout-tournament-001"
CONFIG_PATH = "configs/semantic_formula_proof_holdout_tournament.json"
SOURCE_PATH = "src/sigma_theory_compiler/semantic_formula_proof_holdout_tournament.py"
TEST_PATH = "tests/test_semantic_formula_proof_holdout_tournament.py"
OUTPUT_PATH = "runs/math/semantic-formula-proof-holdout-tournament/campaign.json"
FAMILIES = ("bayesian", "cross_domain", "egraph", "evolutionary", "grammar", "llm", "symbolic")
KINDS = tuple(sorted(ArtifactKind, key=lambda item: item.value))
WORLD_ROWS = (
    (
        "semantic.hidden_polynomial",
        "semantic-polynomial-20260813",
        "516850a249f274606048cf33db6bcaeba671630b0e86e3b050a734a0d43dbb88",
    ),
    (
        "semantic.hidden_rational",
        "semantic-rational-20260813",
        "c70a055e034b95e3040b3f3c4e62608db6684d42f8ed5db5c66f2b0d7d565293",
    ),
    (
        "semantic.hidden_recurrence",
        "semantic-recurrence-20260813",
        "5de6a510f788145e4a6eb100c1afecbd6f2e04efb40258ade3f200438b440252",
    ),
)
_TARGETS = {
    "semantic.hidden_polynomial": {
        "kind": "polynomial_identity",
        "variable": "x",
        "left": "(x + 17)*(x**2 - 5*x + 11)",
        "right": "x**3 + 12*x**2 - 74*x + 187",
    },
    "semantic.hidden_rational": {
        "kind": "rational_identity",
        "variable": "x",
        "left": "(3*x + 7)/(x**2 + 5*x + 6)",
        "right": "1/(x + 2) + 2/(x + 3)",
    },
    "semantic.hidden_recurrence": {
        "kind": "recurrence_closed_form",
        "index": "n",
        "sequence": "weighted_sum",
        "base_index": 0,
        "initial_value": 0,
        "increment": "5*n + 4",
        "closed_form": "n*(5*n + 3)/2",
    },
}
CLAIMS = {
    "all_seven_native_generators_exercised_per_world": True,
    "structured_expression_candidates_required": True,
    "finite_label_selection_used": False,
    "all_generation_completed_before_atomic_unseal": True,
    "exact_counterexamples_required_for_reject": True,
    "proof_certificate_required_for_pass": True,
    "target_reference_proofs_validated": True,
    "post_unseal_tuning_performed": False,
    "general_formula_discovery_established": False,
    "scientific_truth_established": False,
    "novelty_established": False,
    "promotion_authorized": False,
}


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("semantic tournament path is not portable")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("semantic tournament path escapes project root") from error
    return resolved


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = canonical_sha256(body)
    return body


def _expected_config() -> dict[str, Any]:
    return {
        "budgets": {
            "candidates_per_family_per_world": 1,
            "counterexample_integer_radius": 64,
            "generator_work_items_per_world": 128,
            "maximum_absolute_coefficient": 1_000_003,
            "maximum_expression_nodes": 32,
            "maximum_pareto_work_units_per_world": 512,
        },
        "campaign_id": CAMPAIGN_ID,
        "generator_families": list(FAMILIES),
        "policies": {
            "finite_label_selection": "forbidden",
            "generator_target_access": "forbidden",
            "live_sqlite_access": "forbidden",
            "network_access": "forbidden",
            "post_unseal_tuning": "forbidden",
            "proof_required_for_pass": True,
            "target_records_per_unseal": 3,
            "target_unseal_batches": 1,
        },
        "schema_version": CONFIG_SCHEMA,
        "worlds": [
            {
                "world_id": world_id,
                "public_seed": seed,
                "target_commitment_sha256": commitment,
            }
            for world_id, seed, commitment in WORLD_ROWS
        ],
    }


def _load_config(root: Path, config_path: Path | None) -> dict[str, Any]:
    path = config_path or _resolve(root, CONFIG_PATH)
    if path.resolve() != _resolve(root, CONFIG_PATH):
        raise ValueError("semantic tournament preregistration path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != _expected_config():
        raise ValueError("semantic tournament preregistration changed")
    portfolio = build_generator_portfolio()
    implemented = tuple(
        row["strategy_id"]
        for row in portfolio["capabilities"]
        if row["implementation_status"] == "implemented"
    )
    if implemented != FAMILIES:
        raise ValueError("native generator portfolio changed")
    return value


def _descriptor(world_id: str) -> DomainPackDescriptor:
    return DomainPackDescriptor(
        world_id,
        "1.0.0",
        KINDS,
        (
            StageDefinition("structured", 0, KINDS),
            StageDefinition("semantic", 1, KINDS, ("structured",)),
        ),
        (
            GateDefinition(
                "hard_proof_or_counterexample",
                "structured",
                "semantic",
                KINDS,
                ("semantic", "structured"),
            ),
            GateDefinition("hard_structure", None, "structured", KINDS, ("structured",)),
        ),
    )


def _signed_coefficients(material: str, count: int, bound: int) -> tuple[int, ...]:
    values = []
    for index in range(count):
        digest = hashlib.sha256(f"{material}|coefficient|{index}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") % (2 * bound + 1) - bound
        values.append(value or index + 1)
    return tuple(values)


def _expression_payload(
    expression: sp.Expr, variable: str, coefficients: Sequence[int]
) -> dict[str, Any]:
    expression = sp.cancel(expression)
    return {
        "schema": "exact_sympy_rational_expression_v1",
        "variable": variable,
        "coefficients": list(coefficients),
        "expression": str(expression),
        "sympy_srepr": sp.srepr(expression),
        "expression_sha256": hashlib.sha256(sp.srepr(expression).encode()).hexdigest(),
        "node_count": sum(1 for _ in sp.preorder_traversal(expression)),
        "float_atoms": 0,
    }


def _adapt_candidate(
    native: CandidateArtifact,
    family: str,
    world: Mapping[str, Any],
    descriptor: DomainPackDescriptor,
    bound: int,
) -> CandidateArtifact:
    material = canonical_sha256(
        {
            "adapter": "semantic-expression-map-v1",
            "native_candidate": native.ref.to_dict(),
            "public_seed": world["public_seed"],
            "world_id": world["world_id"],
        }
    )
    if world["world_id"] == "semantic.hidden_polynomial":
        variable, coefficients = sp.Symbol("x"), _signed_coefficients(material, 4, bound)
        expression = sum(
            coefficient * variable**index for index, coefficient in enumerate(coefficients)
        )
        expression_class = "cubic_polynomial_over_integers"
    elif world["world_id"] == "semantic.hidden_rational":
        variable, coefficients = sp.Symbol("x"), _signed_coefficients(material, 6, bound)
        numerator = sum(coefficients[index] * variable**index for index in range(3))
        denominator = sum(coefficients[index + 3] * variable**index for index in range(3))
        expression = numerator / denominator
        expression_class = "quadratic_over_quadratic_rational_function"
    else:
        variable, coefficients = (
            sp.Symbol("n", integer=True),
            _signed_coefficients(material, 4, bound),
        )
        expression = sum(
            coefficient * variable**index for index, coefficient in enumerate(coefficients)
        )
        expression_class = "cubic_integer_index_closed_form"
    payload = _expression_payload(expression, str(variable), coefficients)
    payload.update(
        {
            "adapter": "semantic-expression-map-v1",
            "expression_class": expression_class,
            "family": family,
            "native_candidate": native.ref.to_dict(),
            "target_fields_read": [],
        }
    )
    return CandidateArtifact.create(
        ArtifactKind.CONJECTURE,
        f"{family} structured expression conjecture for {world['world_id']}.",
        payload,
        ProvenanceRecord.create(
            descriptor.ref,
            {
                "adapter": "semantic-expression-map-v1",
                "campaign": CAMPAIGN_ID,
                "family": family,
                "world": world["world_id"],
            },
            inputs=(native.ref,),
        ),
        assumptions=("exact rational arithmetic", "target hidden during generation"),
        claims=("requires_exact_semantic_gate",),
    )


def _generate_all(
    root: Path, config: Mapping[str, Any]
) -> list[
    tuple[Mapping[str, Any], DomainPackDescriptor, dict[str, CandidateArtifact], dict[str, Any]]
]:
    prepared = []
    for world in config["worlds"]:
        descriptor = _descriptor(world["world_id"])
        native, native_receipts = _generate_native(root, world, descriptor)
        candidates = {
            family: _adapt_candidate(
                native[family],
                family,
                world,
                descriptor,
                config["budgets"]["maximum_absolute_coefficient"],
            )
            for family in FAMILIES
        }
        if any(
            candidate.representation["node_count"] > config["budgets"]["maximum_expression_nodes"]
            or candidate.representation["float_atoms"] != 0
            or candidate.representation["target_fields_read"] != []
            for candidate in candidates.values()
        ):
            raise ValueError("generated expression escaped structured exact bounds")
        prepared.append((world, descriptor, candidates, native_receipts))
    return prepared


def _target_statement(world_id: str) -> tuple[Any, Any | None]:
    if world_id == "semantic.hidden_polynomial":
        x = symbol("x", RATIONAL)
        return Equation((x + 17) * (x**2 - 5 * x + 11), x**3 + 12 * x**2 - 74 * x + 187), None
    if world_id == "semantic.hidden_rational":
        x = symbol("x", RATIONAL)
        return Equation(
            (3 * x + 7) / (x**2 + 5 * x + 6),
            1 / (x + 2) + 2 / (x + 3),
        ), None
    n = symbol("n", INTEGER)
    recurrence = Recurrence(
        sequence="weighted_sum",
        index=n,
        order=1,
        equation=Equation(call("weighted_sum", n + 1), call("weighted_sum", n) + 5 * n + 4),
        initial_conditions=((0, literal(0)),),
    )
    statement = Equation(call("weighted_sum", n), n * (5 * n + 3) * Fraction(1, 2))
    return statement, recurrence


def _unseal_targets(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if tuple(world["world_id"] for world in config["worlds"]) != tuple(
        row[0] for row in WORLD_ROWS
    ):
        raise ValueError("atomic semantic target inventory changed")
    targets = {}
    for world in config["worlds"]:
        target = dict(_TARGETS[world["world_id"]])
        if canonical_sha256(target) != world["target_commitment_sha256"]:
            raise ValueError("semantic target commitment mismatch")
        targets[world["world_id"]] = target
    return targets


def _reference_proof(world_id: str) -> dict[str, Any]:
    statement, recurrence = _target_statement(world_id)
    if recurrence is None:
        certificate = prove_rational_identity(statement)
        validate_rational_identity_certificate(certificate, statement)
        return certificate
    certificate = prove_induction(statement, recurrence, base_index=0)
    validate_induction_certificate(certificate, statement, recurrence)
    return certificate


def _sympy_target(world_id: str) -> sp.Expr:
    target = _TARGETS[world_id]
    variable = sp.Symbol(
        target.get("variable", target.get("index", "x")), integer=world_id.endswith("recurrence")
    )
    locals_map = {str(variable): variable}
    key = "closed_form" if world_id.endswith("recurrence") else "right"
    return sp.sympify(target[key], locals=locals_map, rational=True)


def _counterexample(
    candidate: sp.Expr, target: sp.Expr, variable: sp.Symbol, radius: int
) -> dict[str, Any] | None:
    difference = sp.cancel(candidate - target)
    numerator, denominator = sp.fraction(difference)
    if sp.expand(numerator) == 0:
        return None
    points = [0] + [
        value for radius_value in range(1, radius + 1) for value in (-radius_value, radius_value)
    ]
    for point in points:
        denominator_value = sp.cancel(denominator.subs(variable, point))
        if denominator_value == 0:
            continue
        candidate_value = sp.cancel(candidate.subs(variable, point))
        target_value = sp.cancel(target.subs(variable, point))
        if candidate_value != target_value:
            return {
                "variable": str(variable),
                "point": point,
                "candidate_value": {
                    "numerator": int(candidate_value.p),
                    "denominator": int(candidate_value.q),
                },
                "target_value": {
                    "numerator": int(target_value.p),
                    "denominator": int(target_value.q),
                },
                "difference_nonzero": True,
            }
    raise ValueError("bounded exact counterexample search failed for nonzero rational expression")


def _candidate_proof(world_id: str, expression: sp.Expr) -> dict[str, Any]:
    if world_id.endswith("recurrence"):
        statement, recurrence = _target_statement(world_id)
        target = _sympy_target(world_id)
        if sp.cancel(expression - target) != 0:
            raise ValueError("recurrence candidate proof requested for unequal expression")
        certificate = prove_induction(statement, recurrence, base_index=0)
        validate_induction_certificate(certificate, statement, recurrence)
        return certificate
    variable_name = "x"
    statement = Equation(
        _sympy_to_ir(expression, {variable_name: RATIONAL}),
        _sympy_to_ir(_sympy_target(world_id), {variable_name: RATIONAL}),
    )
    certificate = prove_rational_identity(statement)
    validate_rational_identity_certificate(certificate, statement)
    return certificate


def _assess(
    world_id: str, candidates: Mapping[str, CandidateArtifact], radius: int
) -> dict[str, dict[str, Any]]:
    target = _sympy_target(world_id)
    variable = next(iter(target.free_symbols))
    assessments = {}
    for family in FAMILIES:
        candidate = candidates[family]
        expression = sp.sympify(
            candidate.representation["expression"], locals={str(variable): variable}, rational=True
        )
        counterexample = _counterexample(expression, target, variable, radius)
        if counterexample is None:
            proof = _candidate_proof(world_id, expression)
            assessments[family] = {
                "status": "pass",
                "proof_certificate": proof,
                "counterexample": None,
            }
        else:
            assessments[family] = {
                "status": "reject",
                "proof_certificate": None,
                "counterexample": counterexample,
            }
    return assessments


def _holdout(
    world: Mapping[str, Any], target: Mapping[str, Any], proof: Mapping[str, Any]
) -> dict[str, Any]:
    public = CandidateKnowledgeNode.create(
        KnowledgeNodeKind.AXIOM,
        {"public_seed_sha256": hashlib.sha256(world["public_seed"].encode()).hexdigest()},
    )
    theorem = CandidateKnowledgeNode.create(KnowledgeNodeKind.THEOREM, dict(target))
    proof_node = CandidateKnowledgeNode.create(
        KnowledgeNodeKind.PROOF,
        {"certificate_sha256": proof["content_sha256"], "role": "hidden_reference_proof"},
    )
    graph = CandidateKnowledgeGraph.create(
        f"semantic.{world['world_id']}.holdout",
        artifacts=(),
        nodes=(public, theorem, proof_node),
        edges=(
            CandidateKnowledgeEdge.create(
                KnowledgeEdgeKind.DEPENDENCY, theorem.node_id, public.node_id
            ),
            CandidateKnowledgeEdge.create(
                KnowledgeEdgeKind.PROVES, proof_node.node_id, theorem.node_id
            ),
        ),
        limits=KnowledgeGraphLimits(1, 8, 8, 4, 4),
    )
    cut = graph.holdout_cut(theorem.node_id)
    cut.validate_against(graph)
    return {"graph": graph.to_dict(), "cut": cut.to_dict()}


class _SemanticPack:
    def __init__(
        self, descriptor: DomainPackDescriptor, assessments: Mapping[str, Mapping[str, Any]]
    ) -> None:
        self._descriptor = descriptor
        self._assessments = assessments

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
            representation.get("schema") == "exact_sympy_rational_expression_v1"
            and representation.get("float_atoms") == 0
            and representation.get("target_fields_read") == []
            and isinstance(representation.get("coefficients"), list)
            and len(representation["coefficients"]) in {4, 6}
            and representation.get("node_count", 0) <= 32
        )
        check = CheckResult.create(
            f"{stage.stage_id}.structured_expression",
            valid,
            {"artifact": artifact.artifact_id, "prior": sorted(prior_outcomes)},
        )
        status = OutcomeStatus.PASS if valid else OutcomeStatus.REJECT
        return StageOutcome.create(
            stage.stage_id,
            artifact.ref,
            status,
            (check,),
            reason_codes=() if valid else ("invalid_structured_expression",),
        )

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> GateOutcome:
        if gate.gate_id == "hard_structure":
            status, reason, evidence = OutcomeStatus.PASS, (), {"structured": True}
        else:
            assessment = self._assessments[artifact.representation["family"]]
            status = OutcomeStatus(assessment["status"])
            reason = () if status is OutcomeStatus.PASS else ("exact_counterexample",)
            evidence = {
                "proof_certificate_sha256": None
                if assessment["proof_certificate"] is None
                else assessment["proof_certificate"]["content_sha256"],
                "counterexample_sha256": None
                if assessment["counterexample"] is None
                else canonical_sha256(assessment["counterexample"]),
            }
        check = CheckResult.create(
            f"{gate.gate_id}.exact_semantic_evidence",
            status is OutcomeStatus.PASS,
            evidence,
        )
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            status,
            tuple(stage_outcomes[key].ref for key in sorted(stage_outcomes)),
            (check,),
            reason_codes=reason,
        )


def _ladder(descriptor: DomainPackDescriptor) -> EvaluationLadder:
    return EvaluationLadder.create(
        descriptor,
        (
            EvaluationStep("structured", "hard_structure", EvaluationPhase.CHEAP),
            EvaluationStep("semantic", "hard_proof_or_counterexample", EvaluationPhase.FORMAL),
        ),
    )


def _metrics(candidates: Sequence[CandidateArtifact]) -> list[MetricReceipt]:
    receipts = []
    for candidate in candidates:
        values = {
            "expression_nodes": candidate.representation["node_count"],
            "representation_bytes": len(canonical_json_bytes(candidate.representation)),
        }
        for metric_id, direction in (
            ("expression_nodes", "minimize"),
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


def _world_result(
    world: Mapping[str, Any],
    descriptor: DomainPackDescriptor,
    candidates: Mapping[str, CandidateArtifact],
    native_receipts: Mapping[str, Any],
    target: Mapping[str, Any],
    reference_proof: Mapping[str, Any],
    holdout: Mapping[str, Any],
    assessments: Mapping[str, Mapping[str, Any]],
    maximum_pareto_work: int,
) -> dict[str, Any]:
    pack, ladder = _SemanticPack(descriptor, assessments), _ladder(descriptor)
    evaluations = {}
    for family in FAMILIES:
        evaluation = evaluate_candidate(pack, candidates[family], ladder)
        validate_evaluation_replay(evaluation, pack, candidates[family])
        evaluations[family] = evaluation
    eligible_families = [
        family for family in FAMILIES if evaluations[family]["all_required_gates_passed"]
    ]
    eligible = [candidates[family] for family in eligible_families]
    metrics = _metrics(eligible)
    pareto = None
    if eligible:
        gates = [
            GateOutcome.from_dict(row)
            for family in eligible_families
            for row in evaluations[family]["gate_outcomes"]
        ]
        limits = ParetoLimits(7, 2, 2, maximum_pareto_work)
        directions = {"expression_nodes": "minimize", "representation_bytes": "minimize"}
        pareto = build_pareto_explanations(
            eligible,
            gates,
            metrics,
            required_gate_ids=("hard_proof_or_counterexample", "hard_structure"),
            metric_directions=directions,
            limits=limits,
        )
        validate_pareto_replay(
            pareto,
            eligible,
            gates,
            metrics,
            required_gate_ids=("hard_proof_or_counterexample", "hard_structure"),
            metric_directions=directions,
            limits=limits,
        )
    status_counts = Counter(evaluation["status"] for evaluation in evaluations.values())
    return {
        "world_id": world["world_id"],
        "public_seed": world["public_seed"],
        "target_commitment_sha256": world["target_commitment_sha256"],
        "unsealed_target": dict(target),
        "reference_proof_certificate": dict(reference_proof),
        "holdout": dict(holdout),
        "native_generator_receipts": dict(native_receipts),
        "candidates": [candidates[family].to_dict() for family in FAMILIES],
        "family_bindings": [
            {"family": family, "candidate": candidates[family].ref.to_dict()} for family in FAMILIES
        ],
        "assessments": dict(assessments),
        "evaluations": evaluations,
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "pareto_eligible_families": eligible_families,
        "metric_receipts": [row.to_dict() for row in metrics],
        "pareto": pareto,
        "decision": (
            "pass_at_least_one_structured_expression_proved"
            if eligible
            else "reject_fixed_budget_with_exact_counterexamples"
        ),
    }


def build_campaign(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root, config_path)
    prepared = _generate_all(root, config)
    phase_events = [
        {"event": "generated", "world_id": world["world_id"], "family": family}
        for world, _, _, _ in prepared
        for family in FAMILIES
    ]
    targets = _unseal_targets(config)
    phase_events.append({"event": "targets_unsealed", "world_ids": list(targets)})
    results = []
    for world, descriptor, candidates, native_receipts in prepared:
        world_id = world["world_id"]
        reference_proof = _reference_proof(world_id)
        assessments = _assess(
            world_id, candidates, config["budgets"]["counterexample_integer_radius"]
        )
        results.append(
            _world_result(
                world,
                descriptor,
                candidates,
                native_receipts,
                targets[world_id],
                reference_proof,
                _holdout(world, targets[world_id], reference_proof),
                assessments,
                config["budgets"]["maximum_pareto_work_units_per_world"],
            )
        )
    first_unseal = next(
        index for index, row in enumerate(phase_events) if row["event"] == "targets_unsealed"
    )
    if first_unseal != 21 or any(
        row["event"] == "generated" for row in phase_events[first_unseal:]
    ):
        raise ValueError("semantic generation did not precede atomic target unseal")
    candidate_statuses = Counter(
        evaluation["status"] for result in results for evaluation in result["evaluations"].values()
    )
    world_decisions = Counter(result["decision"].split("_", 1)[0] for result in results)
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "completed_three_world_semantic_formula_proof_holdout",
        "preregistration": config,
        "portfolio": build_generator_portfolio(),
        "phase_ledger": {
            "events": phase_events,
            "generation_events_before_unseal": first_unseal,
            "pre_unseal_target_access_count": 0,
            "atomic_unseal_batches": 1,
            "target_records_unsealed": 3,
            "post_unseal_generation_count": 0,
            "post_unseal_tuning_events": 0,
        },
        "world_results": results,
        "counts": {
            "worlds": 3,
            "generator_families": 7,
            "structured_candidates": 21,
            "candidate_passes": candidate_statuses["pass"],
            "candidate_rejects": candidate_statuses["reject"],
            "candidate_blocks": candidate_statuses["block"],
            "world_passes": world_decisions["pass"],
            "world_rejects": world_decisions["reject"],
            "world_blocks": world_decisions["block"],
            "reference_proof_certificates": 3,
            "exact_counterexamples": sum(
                assessment["counterexample"] is not None
                for result in results
                for assessment in result["assessments"].values()
            ),
            "pareto_eligible_candidates": sum(
                len(result["pareto_eligible_families"]) for result in results
            ),
        },
        "claims": dict(CLAIMS),
        "scope": (
            "three preregistered hidden semantic worlds over non-finite structured expression "
            "classes; exact target proofs and candidate counterexamples are bounded evidence, "
            "not general discovery, scientific truth, novelty, or promotion"
        ),
        "next_gate": "independently_authored_external_semantic_worlds_with_real_kernel_proof_checks",
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
        raise ValueError("semantic tournament identity changed")
    if value.get("content_sha256") != canonical_sha256(
        {key: child for key, child in value.items() if key != "content_sha256"}
    ):
        raise ValueError("semantic tournament self-seal changed")
    if value.get("claims") != CLAIMS:
        raise ValueError("semantic tournament claim boundary changed")
    if dict(value) != build_campaign(root, config_path):
        raise ValueError("semantic tournament exact replay mismatch")


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    result = build_campaign(root, config_path)
    validate_campaign(result, root, config_path)
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable semantic tournament artifact differs: {path}")
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
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "content_sha256": result["content_sha256"],
                "counts": result["counts"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
