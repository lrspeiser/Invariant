"""Prospective exact constraint-conditioned semantic recovery tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

CONFIG_SCHEMA = "constraint_conditioned_semantic_recovery_tournament_config_v1"
RESULT_SCHEMA = "constraint_conditioned_semantic_recovery_tournament_result_v1"
CAMPAIGN_ID = "constraint-conditioned-semantic-recovery-tournament"
CONFIG_PATH = "configs/constraint_conditioned_semantic_recovery_tournament.json"
SOURCE_PATH = "src/sigma_theory_compiler/constraint_conditioned_semantic_recovery_tournament.py"
TEST_PATH = "tests/test_constraint_conditioned_semantic_recovery_tournament.py"
OUTPUT_PATH = "runs/math/constraint-conditioned-semantic-recovery-tournament/campaign.json"
CONFIG_FILE_SHA256 = "64434d7ec2a78272cf2842219f89910ea77e9f1527846de8153d0a21b3063fb7"
FAMILIES = ("bayesian", "cross_domain", "egraph", "evolutionary", "grammar", "llm", "symbolic")
KINDS = tuple(sorted(ArtifactKind, key=lambda item: item.value))
WORLD_IDS = (
    "constraint.hidden_quartic",
    "constraint.hidden_partial_fraction",
    "constraint.hidden_recurrence",
)
_HIDDEN_TARGETS = {
    "constraint.hidden_quartic": {
        "kind": "polynomial_identity",
        "variable": "x",
        "closed_form": "x**4 + 2*x**3 - x - 30",
    },
    "constraint.hidden_partial_fraction": {
        "kind": "rational_identity",
        "variable": "x",
        "closed_form": "3/(x + 2) - 2/(x + 5) + 5/(x + 7)",
    },
    "constraint.hidden_recurrence": {
        "kind": "recurrence_closed_form",
        "index": "n",
        "sequence": "recovered_sum",
        "base_index": 0,
        "initial_value": 7,
        "increment": "6*n**2 + 10*n + 5",
        "closed_form": "2*n**3 + 2*n**2 + n + 7",
    },
}
CLAIMS = {
    "all_seven_native_generators_exercised_per_world": True,
    "one_generic_target_blind_solver_used": True,
    "all_generation_completed_before_atomic_unseal": True,
    "proof_certificate_required_for_pass": True,
    "exact_malformed_underdetermined_noisy_controls_exercised": True,
    "post_unseal_tuning_performed": False,
    "native_generator_output_establishes_recovery": False,
    "constraint_recovery_establishes_general_formula_discovery": False,
    "scientific_truth_established": False,
    "novelty_established": False,
    "promotion_authorized": False,
}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("constraint recovery path is not portable")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("constraint recovery path escapes project root") from error
    return resolved


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = canonical_sha256(body)
    return body


def _load_config(root: Path, config_path: Path | None) -> dict[str, Any]:
    expected = _resolve(root, CONFIG_PATH)
    path = config_path or expected
    if path.resolve() != expected or _file_sha(path) != CONFIG_FILE_SHA256:
        raise ValueError("constraint recovery preregistration changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or tuple(value.get("generator_families", ())) != FAMILIES
        or tuple(row.get("world_id") for row in value.get("worlds", ())) != WORLD_IDS
        or value.get("preregistration_status") != "sealed_before_generation"
        or len(value.get("controls", ())) != 3
    ):
        raise ValueError("constraint recovery preregistration schema changed")
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
                "hard_proof", "structured", "semantic", KINDS, ("semantic", "structured")
            ),
            GateDefinition("hard_structure", None, "structured", KINDS, ("structured",)),
        ),
    )


def _fraction(value: Mapping[str, Any]) -> sp.Rational:
    if set(value) != {"denominator", "numerator"}:
        raise ValueError("exact rational keys changed")
    numerator, denominator = value["numerator"], value["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ValueError("constraint rational is malformed")
    result = sp.Rational(numerator, denominator)
    if int(result.p) != numerator or int(result.q) != denominator:
        raise ValueError("constraint rational is not normalized")
    return result


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    outcome: str
    reason: str
    rank: int
    augmented_rank: int
    row_count: int
    column_count: int
    row_order: tuple[int, ...]
    coefficients: tuple[sp.Rational, ...] = ()
    expression: sp.Expr | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "generic_exact_linear_synthesis_result_v1",
            "outcome": self.outcome,
            "reason": self.reason,
            "rank": self.rank,
            "augmented_rank": self.augmented_rank,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "row_order": list(self.row_order),
            "coefficients": [
                {"numerator": int(item.p), "denominator": int(item.q)} for item in self.coefficients
            ],
            "expression": None if self.expression is None else str(sp.cancel(self.expression)),
        }
        value["result_sha256"] = canonical_sha256(value)
        return value


def _permutation(size: int, material: str) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(size),
            key=lambda index: hashlib.sha256(f"{material}|row|{index}".encode()).hexdigest(),
        )
    )


def synthesize_from_constraints(
    variable_name: str,
    basis_strings: Sequence[str],
    constraints: Mapping[str, Any],
    *,
    proposal_material: str,
    max_basis_terms: int = 8,
    max_rows: int = 32,
) -> SynthesisResult:
    """Apply one target-blind exact linear solver to a declared basis and constraints."""
    try:
        if (
            not isinstance(variable_name, str)
            or not variable_name.isidentifier()
            or not isinstance(proposal_material, str)
            or not proposal_material
            or not 1 <= len(basis_strings) <= max_basis_terms
        ):
            raise ValueError("malformed synthesis declaration")
        variable = sp.Symbol(variable_name)
        basis = tuple(
            sp.sympify(item, locals={variable_name: variable}, rational=True)
            for item in basis_strings
        )
        if any(expression.free_symbols - {variable} for expression in basis):
            raise ValueError("basis contains an unknown symbol")
        coefficient_symbols = sp.symbols(f"c0:{len(basis)}")
        expression = sp.Add(
            *(
                coefficient * term
                for coefficient, term in zip(coefficient_symbols, basis, strict=True)
            )
        )
        equations: list[sp.Expr] = []
        kind = constraints.get("kind")
        if kind == "evaluations":
            rows = constraints.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError("evaluation constraints are malformed")
            for row in rows:
                if (
                    set(row) != {"point", "value"}
                    or isinstance(row["point"], bool)
                    or not isinstance(row["point"], int)
                ):
                    raise ValueError("evaluation row is malformed")
                equations.append(
                    sp.cancel(expression.subs(variable, row["point"]) - _fraction(row["value"]))
                )
        elif kind == "recurrence":
            if set(constraints) != {"base", "kind", "successor_increment"}:
                raise ValueError("recurrence constraint keys changed")
            base = constraints["base"]
            if (
                set(base) != {"index", "value"}
                or isinstance(base["index"], bool)
                or not isinstance(base["index"], int)
            ):
                raise ValueError("recurrence base is malformed")
            increment = sp.sympify(
                constraints["successor_increment"], locals={variable_name: variable}, rational=True
            )
            if increment.free_symbols - {variable}:
                raise ValueError("recurrence contains an unknown symbol")
            residual = sp.together(expression.subs(variable, variable + 1) - expression - increment)
            numerator, denominator = sp.fraction(residual)
            if denominator != 1:
                raise ValueError("recurrence residual is not polynomial")
            polynomial = sp.Poly(sp.expand(numerator), variable)
            equations.extend(polynomial.all_coeffs())
            equations.append(
                sp.cancel(expression.subs(variable, base["index"]) - _fraction(base["value"]))
            )
        else:
            raise ValueError("constraint kind is unsupported")
        if not equations or len(equations) > max_rows:
            raise ValueError("constraint row budget exceeded")
        rows: list[list[sp.Rational]] = []
        values: list[sp.Rational] = []
        for equation in equations:
            polynomial = sp.Poly(sp.expand(equation), *coefficient_symbols)
            if polynomial.total_degree() > 1:
                raise ValueError("constraint is nonlinear in coefficients")
            rows.append(
                [sp.Rational(polynomial.coeff_monomial(item)) for item in coefficient_symbols]
            )
            values.append(sp.Rational(-polynomial.coeff_monomial(1)))
        order = _permutation(len(rows), proposal_material)
        matrix = sp.Matrix([rows[index] for index in order])
        vector = sp.Matrix([values[index] for index in order])
        rank = int(matrix.rank())
        augmented_rank = int(matrix.row_join(vector).rank())
        if augmented_rank > rank:
            return SynthesisResult(
                "REJECT",
                "inconsistent_exact_constraints",
                rank,
                augmented_rank,
                len(rows),
                len(basis),
                order,
            )
        if rank < len(basis):
            return SynthesisResult(
                "BLOCK",
                "underdetermined_exact_constraints",
                rank,
                augmented_rank,
                len(rows),
                len(basis),
                order,
            )
        solution_set = sp.linsolve((matrix, vector), coefficient_symbols)
        if solution_set is sp.EmptySet or len(solution_set) != 1:
            raise ValueError("unique exact solve failed")
        solution = tuple(sp.Rational(item) for item in next(iter(solution_set)))
        recovered = sp.cancel(
            sum(coefficient * term for coefficient, term in zip(solution, basis, strict=True))
        )
        return SynthesisResult(
            "CANDIDATE",
            "unique_exact_solution",
            rank,
            augmented_rank,
            len(rows),
            len(basis),
            order,
            solution,
            recovered,
        )
    except (KeyError, TypeError, ValueError, sp.SympifyError, sp.PolynomialError) as error:
        return SynthesisResult(
            "BLOCK",
            f"malformed_constraints:{type(error).__name__}",
            0,
            0,
            0,
            len(basis_strings),
            (),
        )


def _expression_payload(
    result: SynthesisResult, family: str, native: CandidateArtifact
) -> dict[str, Any]:
    if result.outcome != "CANDIDATE" or result.expression is None:
        raise ValueError("cannot adapt non-candidate synthesis")
    expression = sp.cancel(result.expression)
    return {
        "schema": "constraint_conditioned_exact_expression_v1",
        "family": family,
        "native_candidate": native.ref.to_dict(),
        "coefficients": result.to_dict()["coefficients"],
        "expression": str(expression),
        "sympy_srepr": sp.srepr(expression),
        "expression_sha256": hashlib.sha256(sp.srepr(expression).encode()).hexdigest(),
        "node_count": sum(1 for _ in sp.preorder_traversal(expression)),
        "solver_receipt": result.to_dict(),
        "target_fields_read": [],
    }


def _generate_all(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    prepared = []
    budgets = config["budgets"]
    for world in config["worlds"]:
        descriptor = _descriptor(world["world_id"])
        native, native_receipts = _generate_native(root, world, descriptor)
        candidates = {}
        for family in FAMILIES:
            parent = native[family]
            material = canonical_sha256(
                {"native": parent.ref.to_dict(), "seed": world["public_seed"]}
            )
            result = synthesize_from_constraints(
                world["variable"],
                world["basis"],
                world["constraints"],
                proposal_material=material,
                max_basis_terms=budgets["max_basis_terms"],
                max_rows=budgets["max_exact_solver_rows"],
            )
            if result.outcome != "CANDIDATE":
                raise ValueError("preregistered world constraints did not uniquely solve")
            payload = _expression_payload(result, family, parent)
            if payload["node_count"] > budgets["max_expression_nodes"]:
                raise ValueError("recovered expression exceeded node budget")
            candidates[family] = CandidateArtifact.create(
                ArtifactKind.CONJECTURE,
                f"{family} lineage with generic exact recovery for {world['world_id']}.",
                payload,
                ProvenanceRecord.create(
                    descriptor.ref,
                    {
                        "campaign": CAMPAIGN_ID,
                        "family": family,
                        "operator": "generic_exact_linear_solve_v1",
                    },
                    inputs=(parent.ref,),
                ),
                assumptions=(
                    "public exact constraints only",
                    "hidden target unavailable during generation",
                ),
                claims=("requires_post_unseal_exact_proof_gate",),
            )
        prepared.append(
            {
                "world": world,
                "descriptor": descriptor,
                "candidates": candidates,
                "native_receipts": native_receipts,
            }
        )
    return prepared


def _unseal_targets(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if tuple(world["world_id"] for world in config["worlds"]) != WORLD_IDS:
        raise ValueError("atomic hidden target inventory changed")
    targets = {}
    for world in config["worlds"]:
        target = dict(_HIDDEN_TARGETS[world["world_id"]])
        if canonical_sha256(target) != world["target_commitment_sha256"]:
            raise ValueError("hidden target commitment mismatch")
        targets[world["world_id"]] = target
    return targets


def _sympy_target(target: Mapping[str, Any]) -> tuple[sp.Symbol, sp.Expr]:
    name = target.get("variable", target.get("index"))
    variable = sp.Symbol(name, integer=target["kind"] == "recurrence_closed_form")
    return variable, sp.sympify(target["closed_form"], locals={name: variable}, rational=True)


def _target_statement(target: Mapping[str, Any]) -> tuple[Equation, Recurrence | None]:
    if target["kind"] != "recurrence_closed_form":
        expression = sp.sympify(
            target["closed_form"],
            locals={target["variable"]: sp.Symbol(target["variable"])},
            rational=True,
        )
        ir = _sympy_to_ir(expression, {target["variable"]: RATIONAL})
        return Equation(ir, ir), None
    n = symbol(target["index"], INTEGER)
    increment = _sympy_to_ir(
        sp.sympify(
            target["increment"], locals={target["index"]: sp.Symbol(target["index"])}, rational=True
        ),
        {target["index"]: INTEGER},
    )
    closed = _sympy_to_ir(
        sp.sympify(
            target["closed_form"],
            locals={target["index"]: sp.Symbol(target["index"])},
            rational=True,
        ),
        {target["index"]: INTEGER},
    )
    recurrence = Recurrence(
        sequence=target["sequence"],
        index=n,
        order=1,
        equation=Equation(call(target["sequence"], n + 1), call(target["sequence"], n) + increment),
        initial_conditions=((target["base_index"], literal(target["initial_value"])),),
    )
    return Equation(call(target["sequence"], n), closed), recurrence


def _proof(target: Mapping[str, Any], expression: sp.Expr) -> dict[str, Any]:
    variable, expected = _sympy_target(target)
    if sp.cancel(expression - expected) != 0:
        raise ValueError("proof requested for unequal recovered expression")
    if target["kind"] == "recurrence_closed_form":
        statement, recurrence = _target_statement(target)
        assert recurrence is not None
        certificate = prove_induction(statement, recurrence, base_index=target["base_index"])
        validate_induction_certificate(certificate, statement, recurrence)
        return certificate
    statement = Equation(
        _sympy_to_ir(expression, {str(variable): RATIONAL}),
        _sympy_to_ir(expected, {str(variable): RATIONAL}),
    )
    certificate = prove_rational_identity(statement)
    validate_rational_identity_certificate(certificate, statement)
    return certificate


def _holdout(
    world: Mapping[str, Any], target: Mapping[str, Any], proof: Mapping[str, Any]
) -> dict[str, Any]:
    public = CandidateKnowledgeNode.create(
        KnowledgeNodeKind.AXIOM,
        {
            "public_constraints_sha256": canonical_sha256(world["constraints"]),
            "public_seed": world["public_seed"],
        },
    )
    theorem = CandidateKnowledgeNode.create(KnowledgeNodeKind.THEOREM, dict(target))
    proof_node = CandidateKnowledgeNode.create(
        KnowledgeNodeKind.PROOF,
        {"certificate_sha256": proof["content_sha256"], "role": "hidden_reference_proof"},
    )
    graph = CandidateKnowledgeGraph.create(
        f"constraint-recovery.{world['world_id']}.holdout",
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


class _RecoveryPack:
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
            set(representation)
            == {
                "coefficients",
                "expression",
                "expression_sha256",
                "family",
                "native_candidate",
                "node_count",
                "schema",
                "solver_receipt",
                "sympy_srepr",
                "target_fields_read",
            }
            and representation["schema"] == "constraint_conditioned_exact_expression_v1"
            and representation["target_fields_read"] == []
            and representation["solver_receipt"]["outcome"] == "CANDIDATE"
            and representation["node_count"] <= 128
        )
        check = CheckResult.create(
            f"{stage.stage_id}.constraint_conditioned_structure",
            valid,
            {"artifact": artifact.artifact_id, "prior": sorted(prior_outcomes)},
        )
        return StageOutcome.create(
            stage.stage_id,
            artifact.ref,
            OutcomeStatus.PASS if valid else OutcomeStatus.REJECT,
            (check,),
            reason_codes=() if valid else ("invalid_constraint_conditioned_expression",),
        )

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> GateOutcome:
        if gate.gate_id == "hard_structure":
            passed = True
            evidence = {"closed_structure": True}
        else:
            assessment = self._assessments[artifact.representation["family"]]
            passed = assessment["status"] == "pass"
            evidence = {
                "proof_certificate_sha256": assessment["proof_certificate"]["content_sha256"]
            }
        check = CheckResult.create(f"{gate.gate_id}.exact_recovery_evidence", passed, evidence)
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            OutcomeStatus.PASS if passed else OutcomeStatus.REJECT,
            tuple(stage_outcomes[key].ref for key in sorted(stage_outcomes)),
            (check,),
            reason_codes=() if passed else ("exact_semantic_mismatch",),
        )


def _ladder(descriptor: DomainPackDescriptor) -> EvaluationLadder:
    return EvaluationLadder.create(
        descriptor,
        (
            EvaluationStep("structured", "hard_structure", EvaluationPhase.CHEAP),
            EvaluationStep("semantic", "hard_proof", EvaluationPhase.FORMAL),
        ),
    )


def _metrics(candidates: Sequence[CandidateArtifact]) -> list[MetricReceipt]:
    rows = []
    for candidate in candidates:
        values = {
            "expression_nodes": candidate.representation["node_count"],
            "representation_bytes": len(canonical_json_bytes(candidate.representation)),
        }
        for metric_id in ("expression_nodes", "representation_bytes"):
            rows.append(
                MetricReceipt.create(
                    candidate.ref,
                    metric_id,
                    "minimize",
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


def _controls(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for control in config["controls"]:
        result = synthesize_from_constraints(
            control["variable"],
            control["basis"],
            control["constraints"],
            proposal_material=control["control_id"],
            max_basis_terms=config["budgets"]["max_basis_terms"],
            max_rows=config["budgets"]["max_exact_solver_rows"],
        )
        if result.outcome != control["expected_outcome"]:
            raise ValueError("negative-control outcome changed")
        rows.append(
            {
                "control_id": control["control_id"],
                "expected_outcome": control["expected_outcome"],
                "observed_outcome": result.outcome,
                "solver_receipt": result.to_dict(),
            }
        )
    return rows


def _world_result(
    prepared: Mapping[str, Any], target: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    world = prepared["world"]
    descriptor = prepared["descriptor"]
    candidates = prepared["candidates"]
    assessments = {}
    for family in FAMILIES:
        target_variable, _ = _sympy_target(target)
        expression = sp.sympify(
            candidates[family].representation["expression"],
            locals={str(target_variable): target_variable},
            rational=True,
        )
        certificate = _proof(target, expression)
        assessments[family] = {"status": "pass", "proof_certificate": certificate}
    reference_proof = _proof(target, _sympy_target(target)[1])
    pack, ladder = _RecoveryPack(descriptor, assessments), _ladder(descriptor)
    evaluations = {}
    gates = []
    eligible = []
    for family in FAMILIES:
        candidate = candidates[family]
        evaluation = evaluate_candidate(pack, candidate, ladder)
        validate_evaluation_replay(evaluation, pack, candidate)
        evaluations[family] = evaluation
        if evaluation["all_required_gates_passed"]:
            eligible.append(candidate)
            gates.extend(GateOutcome.from_dict(row) for row in evaluation["gate_outcomes"])
    metrics = _metrics(eligible)
    limits = ParetoLimits(7, 2, 2, config["budgets"]["max_exact_solver_rows"] * 16)
    directions = {"expression_nodes": "minimize", "representation_bytes": "minimize"}
    pareto = build_pareto_explanations(
        eligible,
        gates,
        metrics,
        required_gate_ids=("hard_proof", "hard_structure"),
        metric_directions=directions,
        limits=limits,
    )
    validate_pareto_replay(
        pareto,
        eligible,
        gates,
        metrics,
        required_gate_ids=("hard_proof", "hard_structure"),
        metric_directions=directions,
        limits=limits,
    )
    return {
        "world_id": world["world_id"],
        "public_seed": world["public_seed"],
        "public_constraints_sha256": canonical_sha256(world["constraints"]),
        "target_commitment_sha256": world["target_commitment_sha256"],
        "unsealed_target": dict(target),
        "reference_proof_certificate": reference_proof,
        "holdout": _holdout(world, target, reference_proof),
        "native_generator_receipts": prepared["native_receipts"],
        "candidates": [candidates[family].to_dict() for family in FAMILIES],
        "assessments": assessments,
        "evaluations": evaluations,
        "terminal_status_counts": {"pass": 7},
        "pareto_eligible_families": list(FAMILIES),
        "metric_receipts": [row.to_dict() for row in metrics],
        "pareto": pareto,
        "decision": "pass_generic_exact_recovery_with_checked_certificate",
    }


def build_campaign(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root, config_path)
    controls = _controls(config)
    prepared = _generate_all(root, config)
    events = [
        {
            "event": "generated_and_constraint_solved",
            "world_id": row["world"]["world_id"],
            "family": family,
        }
        for row in prepared
        for family in FAMILIES
    ]
    targets = _unseal_targets(config)
    events.append({"event": "targets_atomically_unsealed", "world_ids": list(targets)})
    world_results = [
        _world_result(row, targets[row["world"]["world_id"]], config) for row in prepared
    ]
    first_unseal = next(
        index for index, row in enumerate(events) if row["event"] == "targets_atomically_unsealed"
    )
    if first_unseal != 21 or any(
        row["event"] == "generated_and_constraint_solved" for row in events[first_unseal:]
    ):
        raise ValueError("generation did not precede the single atomic unseal")
    statuses = Counter(
        evaluation["status"]
        for result in world_results
        for evaluation in result["evaluations"].values()
    )
    control_counts = Counter(row["observed_outcome"] for row in controls)
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_three_of_three_constraint_conditioned_semantic_worlds",
        "preregistration": config,
        "portfolio": build_generator_portfolio(),
        "phase_ledger": {
            "events": events,
            "generation_and_solve_events_before_unseal": first_unseal,
            "pre_unseal_target_access_count": 0,
            "atomic_unseal_batches": 1,
            "target_records_unsealed": 3,
            "post_unseal_generation_count": 0,
            "post_unseal_tuning_events": 0,
        },
        "control_results": controls,
        "world_results": world_results,
        "counts": {
            "worlds": 3,
            "generator_families": 7,
            "native_generator_invocations": 21,
            "generic_synthesis_invocations": 21,
            "candidate_passes": statuses["pass"],
            "candidate_rejects": statuses["reject"],
            "candidate_blocks": statuses["block"],
            "world_passes": 3,
            "proof_certificates": 21,
            "pareto_eligible_candidates": 21,
            "metric_receipts": 42,
            "control_blocks": control_counts["BLOCK"],
            "control_rejects": control_counts["REJECT"],
        },
        "claims": dict(CLAIMS),
        "scope": (
            "three preregistered hidden targets exactly identifiable in one public declared linear "
            "grammar; recovery attributes to the generic exact solver, while native generators "
            "supply lineage and deterministic proposal material only"
        ),
        "next_gate": "independent_non_linear_or_out_of_basis_semantic_worlds_with_external_kernel_checks",
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
        raise ValueError("constraint recovery campaign identity changed")
    body = {key: child for key, child in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ValueError("constraint recovery campaign self-seal changed")
    if value.get("claims") != CLAIMS:
        raise ValueError("constraint recovery claim boundary changed")
    if dict(value) != build_campaign(root, config_path):
        raise ValueError("constraint recovery exact replay mismatch")


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    result = build_campaign(root, config_path)
    validate_campaign(result, root, config_path)
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable constraint recovery artifact differs: {path}")
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
