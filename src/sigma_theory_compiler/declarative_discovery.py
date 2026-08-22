"""Typed, verifier-separated primitives for open-ended mathematical discovery.

This module is deliberately domain-neutral.  It does not claim that a generated expression is
true or novel.  It defines the protocol by which declarations, creative transformations,
mechanical verification, counterexamples, reachability, proof plans, blind benchmarks, and
dataset explanations can be connected without blurring proposer and verifier roles.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_SCHEMA = "invariant-declarative-discovery-platform-1.0"


class DiscoveryProtocolError(ValueError):
    """The declarative discovery protocol was violated."""


class ValueType(str, Enum):
    SCALAR = "scalar"
    EXPRESSION = "expression"
    SEQUENCE = "sequence"
    EQUATION = "equation"
    INEQUALITY = "inequality"
    PROOF_STATE = "proof_state"
    DATASET_MODEL = "dataset_model"


class DeclarationKind(str, Enum):
    OPERATOR = "operator"
    INVARIANT = "invariant"
    PROOF_TACTIC = "proof_tactic"
    GRAMMAR = "grammar"


class CreativityOperator(str, Enum):
    DIMENSIONAL_ANALYSIS = "dimensional_analysis"
    SYMMETRY_REDUCTION = "symmetry_reduction"
    ANALOGY_TRANSFER = "analogy_transfer"
    DUALITY_TRANSFORM = "duality_transform"
    GENERATING_FUNCTION = "generating_function"
    INVERSE_VARIATIONAL = "inverse_variational"
    RECURRENCE_GUESSING = "recurrence_guessing"
    QUOTIENT_CONSTRUCTION = "quotient_construction"
    CONSERVED_QUANTITY = "conserved_quantity"
    COUNTEREXAMPLE_REPAIR = "counterexample_repair"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class BlockerKind(str, Enum):
    TYPE_MISMATCH = "type_mismatch"
    DOMAIN_HOLE = "domain_hole"
    COUNTEREXAMPLE = "counterexample"
    PROOF_OBLIGATION = "proof_obligation"
    REACHABILITY_GAP = "reachability_gap"
    DATA_INADEQUACY = "data_inadequacy"


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value)
    ):
        raise DiscoveryProtocolError(f"{label} is not a portable identifier")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DiscoveryProtocolError(f"{label} keys changed")


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True, slots=True)
class TypedSymbol:
    name: str
    value_type: ValueType
    dimension: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.name, "symbol name")
        if not isinstance(self.value_type, ValueType):
            raise DiscoveryProtocolError("symbol value_type is not typed")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in self.dimension):
            raise DiscoveryProtocolError("dimension exponents must be integers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": list(self.dimension),
            "name": self.name,
            "value_type": self.value_type.value,
        }


@dataclass(frozen=True, slots=True)
class SearchDeclaration:
    declaration_id: str
    kind: DeclarationKind
    input_types: tuple[ValueType, ...]
    output_type: ValueType
    symbols: tuple[TypedSymbol, ...] = ()
    laws: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.declaration_id, "declaration_id")
        if not isinstance(self.kind, DeclarationKind) or not isinstance(self.output_type, ValueType):
            raise DiscoveryProtocolError("declaration kind and output must be typed")
        if any(not isinstance(item, ValueType) for item in self.input_types):
            raise DiscoveryProtocolError("declaration inputs must be typed")
        names = [item.name for item in self.symbols]
        if len(names) != len(set(names)):
            raise DiscoveryProtocolError("declaration has duplicate symbols")
        if any(not isinstance(law, str) or not law.strip() for law in self.laws):
            raise DiscoveryProtocolError("declaration laws must be nonempty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "declaration_id": self.declaration_id,
            "input_types": [item.value for item in self.input_types],
            "kind": self.kind.value,
            "laws": list(self.laws),
            "output_type": self.output_type.value,
            "symbols": [item.to_dict() for item in self.symbols],
        }


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    operator: CreativityOperator
    declaration_id: str
    input_type: ValueType
    output_type: ValueType
    template: str

    def __post_init__(self) -> None:
        _identifier(self.declaration_id, "operator declaration_id")
        if not isinstance(self.operator, CreativityOperator):
            raise DiscoveryProtocolError("operator is not typed")
        if not isinstance(self.input_type, ValueType) or not isinstance(self.output_type, ValueType):
            raise DiscoveryProtocolError("operator transition is not typed")
        if "{seed}" not in self.template:
            raise DiscoveryProtocolError("operator template must consume {seed}")

    def to_dict(self) -> dict[str, str]:
        return {
            "declaration_id": self.declaration_id,
            "input_type": self.input_type.value,
            "operator": self.operator.value,
            "output_type": self.output_type.value,
            "template": self.template,
        }


DEFAULT_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec(CreativityOperator.DIMENSIONAL_ANALYSIS, "op.dimensionless", ValueType.EXPRESSION, ValueType.EXPRESSION, "dimensionless_basis({seed})"),
    OperatorSpec(CreativityOperator.SYMMETRY_REDUCTION, "op.orbit_average", ValueType.EXPRESSION, ValueType.EXPRESSION, "orbit_average(G, {seed})"),
    OperatorSpec(CreativityOperator.ANALOGY_TRANSFER, "op.analogy", ValueType.EXPRESSION, ValueType.EXPRESSION, "analogy_transport(source, target, {seed})"),
    OperatorSpec(CreativityOperator.DUALITY_TRANSFORM, "op.dual", ValueType.EXPRESSION, ValueType.EXPRESSION, "dual({seed})"),
    OperatorSpec(CreativityOperator.GENERATING_FUNCTION, "op.generating_function", ValueType.SEQUENCE, ValueType.EXPRESSION, "sum(n>=0, ({seed})[n]*z^n)"),
    OperatorSpec(CreativityOperator.INVERSE_VARIATIONAL, "op.inverse_variational", ValueType.EXPRESSION, ValueType.EQUATION, "EulerLagrange(integral({seed}, domain)) = 0"),
    OperatorSpec(CreativityOperator.RECURRENCE_GUESSING, "op.recurrence", ValueType.SEQUENCE, ValueType.EQUATION, "minimal_recurrence({seed}) = 0"),
    OperatorSpec(CreativityOperator.QUOTIENT_CONSTRUCTION, "op.quotient", ValueType.EXPRESSION, ValueType.EXPRESSION, "equivalence_class({seed}, relation)"),
    OperatorSpec(CreativityOperator.CONSERVED_QUANTITY, "op.conservation", ValueType.EXPRESSION, ValueType.EQUATION, "D_t({seed}) = 0"),
    OperatorSpec(CreativityOperator.COUNTEREXAMPLE_REPAIR, "op.counterexample_repair", ValueType.EXPRESSION, ValueType.EXPRESSION, "repair({seed}, witness={witness})"),
)


@dataclass(frozen=True, slots=True)
class Proposal:
    proposal_id: str
    declaration_id: str
    operator: CreativityOperator | None
    value_type: ValueType
    representation: str
    parent_ids: tuple[str, ...]
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.proposal_id, "proposal_id")
        _identifier(self.declaration_id, "proposal declaration_id")
        if self.operator is not None and not isinstance(self.operator, CreativityOperator):
            raise DiscoveryProtocolError("proposal operator must be typed when present")
        if not isinstance(self.value_type, ValueType):
            raise DiscoveryProtocolError("proposal value_type must be typed")
        if not isinstance(self.representation, str) or not self.representation.strip():
            raise DiscoveryProtocolError("proposal representation is empty")
        if not self.parent_ids or any(not isinstance(item, str) or not item for item in self.parent_ids):
            raise DiscoveryProtocolError("proposal must name its parents")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumptions": list(self.assumptions),
            "declaration_id": self.declaration_id,
            "operator": self.operator.value if self.operator else None,
            "parent_ids": list(self.parent_ids),
            "proposal_id": self.proposal_id,
            "representation": self.representation,
            "value_type": self.value_type.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Proposal:
        # Verdict, score, proof, and behaviour fields are intentionally not in this schema.
        _exact_keys(
            value,
            {
                "assumptions",
                "declaration_id",
                "operator",
                "parent_ids",
                "proposal_id",
                "representation",
                "value_type",
            },
            "proposal",
        )
        return cls(
            proposal_id=value["proposal_id"],
            declaration_id=value["declaration_id"],
            operator=(CreativityOperator(value["operator"]) if value["operator"] is not None else None),
            value_type=ValueType(value["value_type"]),
            representation=value["representation"],
            parent_ids=tuple(value["parent_ids"]),
            assumptions=tuple(value["assumptions"]),
        )


@dataclass(frozen=True, slots=True)
class TypedBlocker:
    blocker_id: str
    kind: BlockerKind
    required_type: ValueType
    distance: Fraction
    witness: str
    repair_operator: CreativityOperator

    def __post_init__(self) -> None:
        _identifier(self.blocker_id, "blocker_id")
        if self.distance < 0:
            raise DiscoveryProtocolError("blocker distance cannot be negative")
        if not self.witness:
            raise DiscoveryProtocolError("blocker needs an auditable witness")

    def to_dict(self) -> dict[str, str]:
        return {
            "blocker_id": self.blocker_id,
            "distance": _fraction(self.distance),
            "kind": self.kind.value,
            "repair_operator": self.repair_operator.value,
            "required_type": self.required_type.value,
            "witness": self.witness,
        }


def apply_operator(
    seed: Proposal,
    spec: OperatorSpec,
    *,
    nonce: str,
    blocker: TypedBlocker | None = None,
) -> Proposal:
    """Apply one declared type transition; this function never scores its output."""

    _identifier(nonce, "operator nonce")
    if seed.value_type != spec.input_type:
        raise DiscoveryProtocolError(
            f"{spec.operator.value} needs {spec.input_type.value}, got {seed.value_type.value}"
        )
    if spec.operator is CreativityOperator.COUNTEREXAMPLE_REPAIR:
        if blocker is None or blocker.required_type != seed.value_type:
            raise DiscoveryProtocolError("counterexample repair requires a matching typed blocker")
        witness = blocker.witness
    elif blocker is not None:
        raise DiscoveryProtocolError("only counterexample repair consumes a blocker")
    else:
        witness = "unused"
    representation = spec.template.format(seed=seed.representation, witness=witness)
    identity = canonical_sha256(
        {
            "declaration_id": spec.declaration_id,
            "nonce": nonce,
            "operator": spec.operator.value,
            "parent": seed.proposal_id,
            "representation": representation,
        }
    )
    return Proposal(
        proposal_id=f"proposal-{identity[:24]}",
        declaration_id=spec.declaration_id,
        operator=spec.operator,
        value_type=spec.output_type,
        representation=representation,
        parent_ids=(seed.proposal_id,),
        assumptions=seed.assumptions,
    )


def generate_operator_portfolio(
    seeds: Sequence[Proposal], specs: Sequence[OperatorSpec] = DEFAULT_OPERATORS
) -> tuple[Proposal, ...]:
    """Generate one candidate per compatible creativity family, deterministically."""

    by_type: dict[ValueType, list[Proposal]] = {}
    for seed in sorted(seeds, key=lambda item: item.proposal_id):
        by_type.setdefault(seed.value_type, []).append(seed)
    proposals = []
    for index, spec in enumerate(specs):
        compatible = by_type.get(spec.input_type, [])
        if not compatible or spec.operator is CreativityOperator.COUNTEREXAMPLE_REPAIR:
            continue
        proposals.append(apply_operator(compatible[0], spec, nonce=f"portfolio-{index}"))
    return tuple(proposals)


@dataclass(frozen=True, slots=True)
class BehaviorDescriptor:
    dimensional_signature: tuple[int, ...]
    symmetry_class: str
    complexity_bin: int
    asymptotic_class: str
    invariant_flags: tuple[str, ...]
    singularity_structure: tuple[str, ...] = ()
    conserved_quantities: tuple[str, ...] = ()
    proof_shape: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.complexity_bin < 0 or not self.symmetry_class or not self.asymptotic_class:
            raise DiscoveryProtocolError("behavior descriptor is malformed")
        if tuple(sorted(set(self.invariant_flags))) != self.invariant_flags:
            raise DiscoveryProtocolError("invariant flags must be sorted and unique")
        for label, values in (
            ("singularity structure", self.singularity_structure),
            ("conserved quantities", self.conserved_quantities),
            ("proof shape", self.proof_shape),
        ):
            if tuple(sorted(set(values))) != values or any(not item for item in values):
                raise DiscoveryProtocolError(f"{label} values must be nonempty, sorted, and unique")

    @property
    def niche(self) -> tuple[Any, ...]:
        return (
            self.dimensional_signature,
            self.symmetry_class,
            self.complexity_bin,
            self.asymptotic_class,
            self.invariant_flags,
            self.singularity_structure,
            self.conserved_quantities,
            self.proof_shape,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asymptotic_class": self.asymptotic_class,
            "complexity_bin": self.complexity_bin,
            "dimensional_signature": list(self.dimensional_signature),
            "invariant_flags": list(self.invariant_flags),
            "singularity_structure": list(self.singularity_structure),
            "conserved_quantities": list(self.conserved_quantities),
            "proof_shape": list(self.proof_shape),
            "symmetry_class": self.symmetry_class,
        }


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    proposal_id: str
    verifier_id: str
    status: VerificationStatus
    quality: Fraction
    behavior: BehaviorDescriptor | None
    blockers: tuple[TypedBlocker, ...] = ()
    counterexamples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.proposal_id, "verification proposal_id")
        _identifier(self.verifier_id, "verifier_id")
        if not isinstance(self.quality, Fraction) or not 0 <= self.quality <= 1:
            raise DiscoveryProtocolError("verified quality must be an exact fraction in [0,1]")
        if self.status is VerificationStatus.VERIFIED and (
            self.behavior is None or self.blockers or self.counterexamples
        ):
            raise DiscoveryProtocolError("verified proposal cannot retain blockers")
        if self.status is not VerificationStatus.VERIFIED and not self.blockers:
            raise DiscoveryProtocolError("non-verified proposal requires a typed blocker")

    def to_dict(self) -> dict[str, Any]:
        return {
            "behavior": self.behavior.to_dict() if self.behavior else None,
            "blockers": [item.to_dict() for item in self.blockers],
            "counterexamples": list(self.counterexamples),
            "proposal_id": self.proposal_id,
            "quality": _fraction(self.quality),
            "status": self.status.value,
            "verifier_id": self.verifier_id,
        }


Verifier = Callable[[Proposal], VerificationRecord]


class IndependentVerifierRegistry:
    """Verifier callables live outside proposals and return the only admissible decisions."""

    def __init__(self) -> None:
        self._verifiers: dict[ValueType, tuple[str, Verifier]] = {}

    def register(self, value_type: ValueType, verifier_id: str, verifier: Verifier) -> None:
        _identifier(verifier_id, "verifier_id")
        if value_type in self._verifiers or not callable(verifier):
            raise DiscoveryProtocolError("verifier registration is duplicate or not callable")
        self._verifiers[value_type] = verifier_id, verifier

    def verify(self, proposal: Proposal) -> VerificationRecord:
        if proposal.value_type not in self._verifiers:
            raise DiscoveryProtocolError(f"no verifier for {proposal.value_type.value}")
        verifier_id, verifier = self._verifiers[proposal.value_type]
        record = verifier(proposal)
        if record.proposal_id != proposal.proposal_id or record.verifier_id != verifier_id:
            raise DiscoveryProtocolError("verifier returned a record with changed provenance")
        return record


class BehavioralMapElites:
    """One best independently verified proposal per measured behavioral niche."""

    def __init__(self) -> None:
        self._cells: dict[tuple[Any, ...], tuple[Proposal, VerificationRecord]] = {}

    def insert(self, proposal: Proposal, record: VerificationRecord) -> bool:
        if record.proposal_id != proposal.proposal_id or record.status is not VerificationStatus.VERIFIED:
            raise DiscoveryProtocolError("MAP-Elites admits independently verified proposals only")
        assert record.behavior is not None
        key = record.behavior.niche
        incumbent = self._cells.get(key)
        if incumbent is not None:
            better = record.quality > incumbent[1].quality or (
                record.quality == incumbent[1].quality
                and proposal.proposal_id < incumbent[0].proposal_id
            )
            if not better:
                return False
        self._cells[key] = proposal, record
        return True

    @property
    def occupied_niches(self) -> int:
        return len(self._cells)

    def elites(self) -> tuple[Proposal, ...]:
        return tuple(self._cells[key][0] for key in sorted(self._cells, key=repr))


@dataclass(frozen=True, slots=True)
class ReachabilityCertificate:
    initial_type: ValueType
    target_type: ValueType
    operator_path: tuple[CreativityOperator, ...]
    witness_proposal_id: str

    def validate(self, specs: Sequence[OperatorSpec] = DEFAULT_OPERATORS) -> None:
        _identifier(self.witness_proposal_id, "reachability witness")
        current = self.initial_type
        by_operator = {item.operator: item for item in specs}
        for operator in self.operator_path:
            spec = by_operator.get(operator)
            if spec is None or spec.input_type != current:
                raise DiscoveryProtocolError("reachability path contains an invalid type transition")
            current = spec.output_type
        if current != self.target_type:
            raise DiscoveryProtocolError("reachability path does not reach the target type")


def find_type_reachability(
    initial_type: ValueType,
    target_type: ValueType,
    witness_proposal_id: str,
    specs: Sequence[OperatorSpec] = DEFAULT_OPERATORS,
) -> ReachabilityCertificate:
    queue: deque[tuple[ValueType, tuple[CreativityOperator, ...]]] = deque([(initial_type, ())])
    seen = {initial_type}
    while queue:
        current, path = queue.popleft()
        if current == target_type:
            result = ReachabilityCertificate(initial_type, target_type, path, witness_proposal_id)
            result.validate(specs)
            return result
        for spec in specs:
            if spec.input_type == current and spec.output_type not in seen:
                seen.add(spec.output_type)
                queue.append((spec.output_type, (*path, spec.operator)))
    raise DiscoveryProtocolError("target type is unreachable in the declared operator graph")


@dataclass(frozen=True, slots=True)
class NegativeResult:
    target_id: str
    explored_proposals: int
    reachability: ReachabilityCertificate
    status: str = "REAL_NEGATIVE"


def publish_negative(
    target_id: str, explored_proposals: int, reachability: ReachabilityCertificate
) -> NegativeResult:
    _identifier(target_id, "negative target_id")
    if explored_proposals < 1:
        raise DiscoveryProtocolError("negative result requires a nonempty search")
    reachability.validate()
    return NegativeResult(target_id, explored_proposals, reachability)


@dataclass(frozen=True, slots=True)
class TacticDeclaration:
    tactic_id: str
    consumes: str
    produces: tuple[str, ...]
    cost: int = 1

    def __post_init__(self) -> None:
        _identifier(self.tactic_id, "tactic_id")
        _identifier(self.consumes, "tactic goal kind")
        if self.cost < 1 or any(not isinstance(item, str) or not item for item in self.produces):
            raise DiscoveryProtocolError("tactic declaration is malformed")


@dataclass(frozen=True, slots=True)
class ProofPlan:
    proposal_id: str
    tactic_ids: tuple[str, ...]
    closed: bool
    remaining_goals: tuple[str, ...]


def search_proof_plan(
    proposal_id: str,
    initial_goals: Sequence[str],
    tactics: Sequence[TacticDeclaration],
    *,
    max_steps: int = 12,
) -> ProofPlan:
    """Breadth-first proof-plan search over declared goal-shape transitions."""

    _identifier(proposal_id, "proof-plan proposal_id")
    start = tuple(sorted(initial_goals))
    queue: deque[tuple[tuple[str, ...], tuple[str, ...]]] = deque([(start, ())])
    best_depth = {start: 0}
    ordered = tuple(sorted(tactics, key=lambda item: (item.cost, item.tactic_id)))
    while queue:
        goals, plan = queue.popleft()
        if not goals:
            return ProofPlan(proposal_id, plan, True, ())
        if len(plan) >= max_steps:
            continue
        goal = goals[0]
        for tactic in ordered:
            if tactic.consumes != goal:
                continue
            next_goals = tuple(sorted((*goals[1:], *tactic.produces)))
            next_plan = (*plan, tactic.tactic_id)
            if best_depth.get(next_goals, max_steps + 1) <= len(next_plan):
                continue
            best_depth[next_goals] = len(next_plan)
            queue.append((next_goals, next_plan))
    return ProofPlan(proposal_id, (), False, start)


class DatasetStage(str, Enum):
    SHAPE_AUDIT = "shape_audit"
    UNIT_NORMALIZATION = "unit_normalization"
    TRAIN_HOLDOUT_SPLIT = "train_holdout_split"
    CANDIDATE_FIT = "candidate_fit"
    RESIDUAL_PROBE = "residual_probe"
    HELDOUT_TEST = "heldout_test"
    MECHANISM_FALSIFIER = "mechanism_falsifier"
    EXPLANATION = "explanation"


DATASET_STAGES = tuple(DatasetStage)


@dataclass(frozen=True, slots=True)
class DatasetStageRecord:
    stage: DatasetStage
    input_sha256: str
    output_sha256: str
    passed: bool
    heldout_opened: bool


class DatasetExplanationPipeline:
    """Fail-closed stage machine that keeps held-out data sealed until its test."""

    def __init__(self, dataset_sha256: str, heldout_commitment: str) -> None:
        self.dataset_sha256 = dataset_sha256
        self.heldout_commitment = heldout_commitment
        self.records: list[DatasetStageRecord] = []

    def record(
        self,
        stage: DatasetStage,
        input_sha256: str,
        output_sha256: str,
        *,
        passed: bool,
        heldout_opened: bool = False,
    ) -> DatasetStageRecord:
        if self.records and not self.records[-1].passed:
            raise DiscoveryProtocolError("dataset pipeline cannot continue after a failed stage")
        expected = DATASET_STAGES[len(self.records)] if len(self.records) < len(DATASET_STAGES) else None
        if stage is not expected:
            raise DiscoveryProtocolError("dataset pipeline stage is missing or out of order")
        stage_index = DATASET_STAGES.index(stage)
        heldout_index = DATASET_STAGES.index(DatasetStage.HELDOUT_TEST)
        if stage_index < heldout_index and heldout_opened:
            raise DiscoveryProtocolError("held-out data opened before the held-out test")
        if stage is DatasetStage.HELDOUT_TEST and not heldout_opened:
            raise DiscoveryProtocolError("held-out test did not open its committed target")
        row = DatasetStageRecord(stage, input_sha256, output_sha256, passed, heldout_opened)
        self.records.append(row)
        return row

    @property
    def completed(self) -> bool:
        return len(self.records) == len(DATASET_STAGES) and all(item.passed for item in self.records)


class CapabilityLevel(int, Enum):
    SOLVED_VISIBLE = 1
    SOLVED_ANONYMOUS = 2
    SYNTHETIC_TARGET_SEALED = 3
    HISTORICAL_TARGET_SEALED = 4
    OUT_OF_DISTRIBUTION = 5
    OPEN_PROBLEM = 6


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    level: CapabilityLevel
    benchmark_id: str
    target_commitment: str
    target_opened_after_proposal: bool
    leakage_tokens: tuple[str, ...]
    passed: bool


class BlindCapabilityLadder:
    def __init__(self) -> None:
        self.results: list[CapabilityResult] = []

    def admit(self, result: CapabilityResult) -> None:
        expected = len(self.results) + 1
        if result.level.value != expected:
            raise DiscoveryProtocolError("blind capability levels cannot be skipped")
        if result.level.value >= CapabilityLevel.SYNTHETIC_TARGET_SEALED.value and (
            not result.target_commitment
            or not result.target_opened_after_proposal
            or result.leakage_tokens
        ):
            raise DiscoveryProtocolError("sealed blind level leaked or opened its target early")
        if self.results and not self.results[-1].passed:
            raise DiscoveryProtocolError("capability ladder cannot advance past a failed level")
        self.results.append(result)

    @property
    def highest_passed(self) -> int:
        return max((item.level.value for item in self.results if item.passed), default=0)


class ChainStage(int, Enum):
    DECLARATION = 1
    PROPOSAL = 2
    VERIFICATION = 3
    COUNTEREXAMPLE = 4
    REPAIR = 5
    PROOF_PLAN = 6
    BLIND_BENCHMARK = 7
    DATASET_EXPLANATION = 8
    PRIOR_ART = 9
    RELEASE = 10


@dataclass(frozen=True, slots=True)
class DiscoveryChainLink:
    link_id: str
    stage: ChainStage
    artifact_sha256: str
    parent_ids: tuple[str, ...]


class DiscoveryChain:
    """Typed provenance DAG from declaration through independent release evidence."""

    def __init__(self) -> None:
        self.links: dict[str, DiscoveryChainLink] = {}

    def add(self, link: DiscoveryChainLink) -> None:
        _identifier(link.link_id, "chain link_id")
        if link.link_id in self.links:
            raise DiscoveryProtocolError("duplicate discovery-chain link")
        if link.stage is ChainStage.DECLARATION:
            if link.parent_ids:
                raise DiscoveryProtocolError("declaration link cannot have parents")
        elif not link.parent_ids:
            raise DiscoveryProtocolError("non-declaration link requires provenance parents")
        for parent_id in link.parent_ids:
            parent = self.links.get(parent_id)
            if parent is None or parent.stage.value >= link.stage.value:
                raise DiscoveryProtocolError("chain parent is absent or not an earlier stage")
        self.links[link.link_id] = link

    def content_sha256(self) -> str:
        return canonical_sha256(
            [
                {
                    "artifact_sha256": link.artifact_sha256,
                    "link_id": link.link_id,
                    "parent_ids": list(link.parent_ids),
                    "stage": link.stage.name.lower(),
                }
                for link in sorted(self.links.values(), key=lambda item: item.link_id)
            ]
        )


@dataclass(frozen=True, slots=True)
class CreativeYieldMetrics:
    proposals: int
    verified: int
    behavioral_niches: int
    proof_plans_closed: int
    counterexample_repairs_attempted: int
    counterexample_repairs_verified: int
    blind_levels_passed: int
    dataset_explanations_completed: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values):
            raise DiscoveryProtocolError("creative-yield counts must be nonnegative integers")
        if self.verified > self.proposals or self.behavioral_niches > self.verified:
            raise DiscoveryProtocolError("creative-yield counts are inconsistent")
        if self.counterexample_repairs_verified > self.counterexample_repairs_attempted:
            raise DiscoveryProtocolError("repair successes exceed repair attempts")

    def to_dict(self) -> dict[str, Any]:
        def ratio(numerator: int, denominator: int) -> str:
            return _fraction(Fraction(numerator, denominator)) if denominator else "0/1"

        return {
            "counts": {name: getattr(self, name) for name in self.__dataclass_fields__},
            "rates": {
                "behavioral_yield_per_proposal": ratio(self.behavioral_niches, self.proposals),
                "repair_success": ratio(
                    self.counterexample_repairs_verified,
                    self.counterexample_repairs_attempted,
                ),
                "verification_yield": ratio(self.verified, self.proposals),
            },
            "claims": {
                "behavioral_difference_measured": True,
                "novelty_established": False,
                "truth_established_by_yield_metric": False,
            },
        }


def measure_creative_yield(
    proposals: Sequence[Proposal],
    verifications: Sequence[VerificationRecord],
    archive: BehavioralMapElites,
    proof_plans: Sequence[ProofPlan],
    *,
    repairs_attempted: int,
    repairs_verified: int,
    blind_levels_passed: int,
    dataset_explanations_completed: int,
) -> CreativeYieldMetrics:
    verified = sum(item.status is VerificationStatus.VERIFIED for item in verifications)
    return CreativeYieldMetrics(
        proposals=len(proposals),
        verified=verified,
        behavioral_niches=archive.occupied_niches,
        proof_plans_closed=sum(item.closed for item in proof_plans),
        counterexample_repairs_attempted=repairs_attempted,
        counterexample_repairs_verified=repairs_verified,
        blind_levels_passed=blind_levels_passed,
        dataset_explanations_completed=dataset_explanations_completed,
    )


@dataclass(frozen=True, slots=True)
class PlatformConfig:
    config_id: str
    operators: tuple[CreativityOperator, ...]
    dataset_stages: tuple[DatasetStage, ...]
    capability_levels: tuple[CapabilityLevel, ...]
    maximum_proposals: int
    proof_plan_max_steps: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PlatformConfig:
        _exact_keys(
            value,
            {
                "capability_levels",
                "config_id",
                "dataset_stages",
                "limits",
                "operators",
                "schema_version",
            },
            "platform config",
        )
        if value["schema_version"] != CONFIG_SCHEMA:
            raise DiscoveryProtocolError("platform config schema changed")
        limits = value["limits"]
        _exact_keys(limits, {"maximum_proposals", "proof_plan_max_steps"}, "platform limits")
        result = cls(
            config_id=_identifier(value["config_id"], "config_id"),
            operators=tuple(CreativityOperator(item) for item in value["operators"]),
            dataset_stages=tuple(DatasetStage(item) for item in value["dataset_stages"]),
            capability_levels=tuple(CapabilityLevel(item) for item in value["capability_levels"]),
            maximum_proposals=limits["maximum_proposals"],
            proof_plan_max_steps=limits["proof_plan_max_steps"],
        )
        if (
            result.operators != tuple(CreativityOperator)
            or result.dataset_stages != DATASET_STAGES
            or result.capability_levels != tuple(CapabilityLevel)
            or result.maximum_proposals < 1
            or result.proof_plan_max_steps < 1
        ):
            raise DiscoveryProtocolError("platform config omits a required lane or has invalid limits")
        return result


def load_platform_config(path: str | Path) -> PlatformConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiscoveryProtocolError("platform config is unavailable") from error
    if not isinstance(value, Mapping):
        raise DiscoveryProtocolError("platform config must be an object")
    return PlatformConfig.from_dict(value)
