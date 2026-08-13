"""Bounded legacy-grammar adapter producing Sigma Core candidate artifacts.

The adapter preserves the node-cost and SymPy canonicalization semantics of
``grammar.enumerate_expressions`` while adding closed registries, hard resource caps, exact
provenance, and replayable lineage.  Generation is syntactic only: neither a successful run nor
an emitted formula establishes truth, physical validity, or promotion eligibility.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import sympy as sp

from .grammar import SYMBOLS, canonicalize
from .sigma_core import (
    ArtifactKind,
    ArtifactRef,
    CandidateArtifact,
    DomainPackRef,
    OutcomeStatus,
    ProvenanceRecord,
    SchemaViolation,
    SourceBinding,
    canonical_sha256,
)

SCHEMA_VERSION = "sigma-grammar-candidate-generator-1.0"
LINEAGE_SCHEMA_VERSION = "sigma-grammar-lineage-1.0"
REGISTERED_ATOMS = tuple(sorted(SYMBOLS))
REGISTERED_UNARY_OPERATORS = ("saturate", "sqrt1p_minus1")
REGISTERED_BINARY_OPERATORS = ("add", "multiply")

# These module ceilings cannot be raised through caller input.
HARD_MAXIMUM_COMPLEXITY = 9
HARD_MAXIMUM_EXPRESSIONS = 25_000
HARD_MAXIMUM_TOTAL_NODES = 250_000
HARD_MAXIMUM_WORK_UNITS = 250_000

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SOURCE_PATHS = {
    "adapter": "src/sigma_theory_compiler/grammar_candidate_generator.py",
    "legacy_grammar": "src/sigma_theory_compiler/grammar.py",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SchemaViolation(f"{label} keys changed")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} is not a canonical identifier")
    return value


def _strict_positive(value: Any, label: str, ceiling: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
        raise SchemaViolation(f"{label} must be an integer in [1, {ceiling}]")
    return value


def _closed_names(
    values: Sequence[str], label: str, *, require_nonempty: bool = False
) -> tuple[str, ...]:
    result = tuple(_identifier(item, label) for item in values)
    if (require_nonempty and not result) or result != tuple(sorted(set(result))):
        qualifier = "nonempty, " if require_nonempty else ""
        raise SchemaViolation(f"{label} values must be {qualifier}unique and sorted")
    return result


def _reason_codes(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_identifier(item, "reason code") for item in values)
    if not result or result != tuple(sorted(set(result))):
        raise SchemaViolation("reason codes must be nonempty, unique, and sorted")
    return result


@dataclass(frozen=True, slots=True)
class GrammarSpec:
    """Closed declaration of the grammar searched by one run.

    Names are structurally validated here but registry membership is evaluated by generation so
    unknown names produce a typed ``reject`` manifest instead of an untyped exception.
    """

    atoms: tuple[str, ...]
    unary_operators: tuple[str, ...]
    binary_operators: tuple[str, ...]
    maximum_complexity: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", _closed_names(self.atoms, "atom", require_nonempty=True))
        object.__setattr__(
            self,
            "unary_operators",
            _closed_names(self.unary_operators, "unary operator"),
        )
        object.__setattr__(
            self,
            "binary_operators",
            _closed_names(self.binary_operators, "binary operator"),
        )
        _strict_positive(
            self.maximum_complexity,
            "maximum_complexity",
            HARD_MAXIMUM_COMPLEXITY,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atoms": list(self.atoms),
            "unary_operators": list(self.unary_operators),
            "binary_operators": list(self.binary_operators),
            "maximum_complexity": self.maximum_complexity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GrammarSpec:
        _exact_keys(
            value,
            {"atoms", "unary_operators", "binary_operators", "maximum_complexity"},
            "grammar spec",
        )
        if not all(
            isinstance(value[key], list) for key in ("atoms", "unary_operators", "binary_operators")
        ):
            raise SchemaViolation("grammar name collections must be arrays")
        return cls(
            tuple(value["atoms"]),
            tuple(value["unary_operators"]),
            tuple(value["binary_operators"]),
            value["maximum_complexity"],
        )


@dataclass(frozen=True, slots=True)
class GrammarLimits:
    """Non-negotiable expression, aggregate-node, and work budgets."""

    maximum_expressions: int
    maximum_total_nodes: int
    maximum_work_units: int

    def __post_init__(self) -> None:
        _strict_positive(
            self.maximum_expressions,
            "maximum_expressions",
            HARD_MAXIMUM_EXPRESSIONS,
        )
        _strict_positive(
            self.maximum_total_nodes,
            "maximum_total_nodes",
            HARD_MAXIMUM_TOTAL_NODES,
        )
        _strict_positive(
            self.maximum_work_units,
            "maximum_work_units",
            HARD_MAXIMUM_WORK_UNITS,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_expressions": self.maximum_expressions,
            "maximum_total_nodes": self.maximum_total_nodes,
            "maximum_work_units": self.maximum_work_units,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GrammarLimits:
        _exact_keys(
            value,
            {"maximum_expressions", "maximum_total_nodes", "maximum_work_units"},
            "grammar limits",
        )
        return cls(
            value["maximum_expressions"],
            value["maximum_total_nodes"],
            value["maximum_work_units"],
        )


@dataclass(frozen=True, slots=True)
class GrammarCounts:
    work_units: int
    unique_discovered: int
    duplicates_observed: int
    cap_rejected_unique: int
    failed_work_units: int
    emitted_expressions: int
    emitted_nodes: int

    def __post_init__(self) -> None:
        for name in (
            "work_units",
            "unique_discovered",
            "duplicates_observed",
            "cap_rejected_unique",
            "failed_work_units",
            "emitted_expressions",
            "emitted_nodes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaViolation(f"{name} must be a nonnegative integer")
        if self.work_units != (
            self.unique_discovered
            + self.duplicates_observed
            + self.cap_rejected_unique
            + self.failed_work_units
        ):
            raise SchemaViolation("grammar work accounting changed")

    def to_dict(self) -> dict[str, int]:
        return {
            "work_units": self.work_units,
            "unique_discovered": self.unique_discovered,
            "duplicates_observed": self.duplicates_observed,
            "cap_rejected_unique": self.cap_rejected_unique,
            "failed_work_units": self.failed_work_units,
            "emitted_expressions": self.emitted_expressions,
            "emitted_nodes": self.emitted_nodes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GrammarCounts:
        expected = {
            "work_units",
            "unique_discovered",
            "duplicates_observed",
            "cap_rejected_unique",
            "failed_work_units",
            "emitted_expressions",
            "emitted_nodes",
        }
        _exact_keys(value, expected, "grammar counts")
        return cls(*(value[key] for key in expected_order()))


def expected_order() -> tuple[str, ...]:
    return (
        "work_units",
        "unique_discovered",
        "duplicates_observed",
        "cap_rejected_unique",
        "failed_work_units",
        "emitted_expressions",
        "emitted_nodes",
    )


@dataclass(frozen=True, slots=True)
class GrammarLineage:
    ordinal: int
    operation: str
    operator: str
    child: ArtifactRef
    parents: tuple[ArtifactRef, ...]
    complexity: int
    lineage_sha256: str
    schema_version: str = LINEAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LINEAGE_SCHEMA_VERSION:
            raise SchemaViolation("grammar lineage schema_version changed")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise SchemaViolation("grammar lineage ordinal must be a nonnegative integer")
        if self.operation not in {"atom", "unary", "binary"}:
            raise SchemaViolation("grammar lineage operation is not registered")
        _identifier(self.operator, "lineage operator")
        _strict_positive(self.complexity, "lineage complexity", HARD_MAXIMUM_COMPLEXITY)
        expected_parents = {"atom": 0, "unary": 1, "binary": 2}[self.operation]
        if len(self.parents) != expected_parents:
            raise SchemaViolation("grammar lineage parent count changed")
        if self.parents != tuple(sorted(self.parents, key=lambda item: item.artifact_id)):
            raise SchemaViolation("grammar lineage parents must be sorted")
        if self.lineage_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("grammar lineage canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "operation": self.operation,
            "operator": self.operator,
            "child": self.child.to_dict(),
            "parents": [item.to_dict() for item in self.parents],
            "complexity": self.complexity,
        }

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        operation: str,
        operator: str,
        child: ArtifactRef,
        parents: Sequence[ArtifactRef],
        complexity: int,
    ) -> GrammarLineage:
        ordered = tuple(sorted(parents, key=lambda item: item.artifact_id))
        body = {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "ordinal": ordinal,
            "operation": operation,
            "operator": operator,
            "child": child.to_dict(),
            "parents": [item.to_dict() for item in ordered],
            "complexity": complexity,
        }
        return cls(
            ordinal,
            operation,
            operator,
            child,
            ordered,
            complexity,
            canonical_sha256(body),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "lineage_sha256": self.lineage_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GrammarLineage:
        _exact_keys(
            value,
            {
                "schema_version",
                "ordinal",
                "operation",
                "operator",
                "child",
                "parents",
                "complexity",
                "lineage_sha256",
            },
            "grammar lineage",
        )
        if not isinstance(value["parents"], list):
            raise SchemaViolation("grammar lineage parents must be an array")
        return cls(
            value["ordinal"],
            str(value["operation"]),
            str(value["operator"]),
            ArtifactRef.from_dict(value["child"]),
            tuple(ArtifactRef.from_dict(item) for item in value["parents"]),
            value["complexity"],
            str(value["lineage_sha256"]),
            str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class GrammarGenerationManifest:
    spec: GrammarSpec
    limits: GrammarLimits
    domain_pack: DomainPackRef
    sources: tuple[SourceBinding, ...]
    status: OutcomeStatus
    reason_codes: tuple[str, ...]
    counts: GrammarCounts
    candidates: tuple[CandidateArtifact, ...]
    lineage: tuple[GrammarLineage, ...]
    manifest_sha256: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaViolation("grammar manifest schema_version changed")
        if self.sources != tuple(sorted(self.sources, key=lambda item: item.role)):
            raise SchemaViolation("grammar manifest sources must be sorted")
        if {item.role for item in self.sources} != set(_SOURCE_PATHS):
            raise SchemaViolation("grammar manifest source roles changed")
        if self.status is OutcomeStatus.PASS:
            if self.reason_codes:
                raise SchemaViolation("pass grammar manifest cannot carry reason codes")
            if not self.candidates:
                raise SchemaViolation("pass grammar manifest must emit candidates")
        else:
            _reason_codes(self.reason_codes)
            if self.candidates or self.lineage:
                raise SchemaViolation("non-pass grammar manifest cannot emit partial results")
        if len(self.candidates) != len(self.lineage):
            raise SchemaViolation("grammar candidate and lineage counts differ")
        if self.counts.emitted_expressions != len(self.candidates):
            raise SchemaViolation("grammar emitted expression count changed")
        if self.counts.emitted_nodes != sum(
            item.representation["complexity"] for item in self.candidates
        ):
            raise SchemaViolation("grammar emitted node count changed")
        if self.status is not OutcomeStatus.PASS and (
            self.counts.emitted_expressions != 0 or self.counts.emitted_nodes != 0
        ):
            raise SchemaViolation("non-pass grammar manifest emitted counts must be zero")
        candidate_refs = {item.artifact_id: item.ref for item in self.candidates}
        if len(candidate_refs) != len(self.candidates):
            raise SchemaViolation("grammar candidates contain duplicate artifact IDs")
        if tuple(item.ordinal for item in self.lineage) != tuple(range(len(self.lineage))):
            raise SchemaViolation("grammar lineage ordinals are not contiguous")
        for candidate, record in zip(self.candidates, self.lineage, strict=True):
            candidate.validate()
            if record.child != candidate.ref:
                raise SchemaViolation("grammar lineage child does not match candidate order")
            if any(parent.artifact_id not in candidate_refs for parent in record.parents):
                raise SchemaViolation("grammar lineage parent is outside the manifest")
        if self.manifest_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("grammar manifest canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "limits": self.limits.to_dict(),
            "domain_pack": self.domain_pack.to_dict(),
            "sources": [item.to_dict() for item in self.sources],
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "counts": self.counts.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "lineage": [item.to_dict() for item in self.lineage],
            "generation_only": True,
            "truth_established": False,
            "promotion_allowed": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "manifest_sha256": self.manifest_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GrammarGenerationManifest:
        _exact_keys(
            value,
            {
                "schema_version",
                "spec",
                "limits",
                "domain_pack",
                "sources",
                "status",
                "reason_codes",
                "counts",
                "candidates",
                "lineage",
                "generation_only",
                "truth_established",
                "promotion_allowed",
                "manifest_sha256",
            },
            "grammar manifest",
        )
        if (
            value["generation_only"] is not True
            or value["truth_established"] is not False
            or value["promotion_allowed"] is not False
        ):
            raise SchemaViolation("grammar manifest scientific boundary changed")
        for key in ("sources", "reason_codes", "candidates", "lineage"):
            if not isinstance(value[key], list):
                raise SchemaViolation(f"grammar manifest {key} must be an array")
        try:
            status = OutcomeStatus(value["status"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("grammar manifest status is not registered") from error
        return cls(
            GrammarSpec.from_dict(value["spec"]),
            GrammarLimits.from_dict(value["limits"]),
            DomainPackRef.from_dict(value["domain_pack"]),
            tuple(SourceBinding.from_dict(item) for item in value["sources"]),
            status,
            tuple(value["reason_codes"]),
            GrammarCounts.from_dict(value["counts"]),
            tuple(CandidateArtifact.from_dict(item) for item in value["candidates"]),
            tuple(GrammarLineage.from_dict(item) for item in value["lineage"]),
            str(value["manifest_sha256"]),
            str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class _Generated:
    expression: sp.Expr
    complexity: int
    canonical: str
    operation: str
    operator: str
    parent_canonicals: tuple[str, ...]


class _CapReached(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(slots=True)
class _Counter:
    limits: GrammarLimits
    work_units: int = 0
    unique_discovered: int = 0
    duplicates_observed: int = 0
    cap_rejected_unique: int = 0
    failed_work_units: int = 0
    discovered_nodes: int = 0

    def before_attempt(self) -> None:
        if self.work_units >= self.limits.maximum_work_units:
            raise _CapReached("work_cap_reached")
        self.work_units += 1

    def before_unique(self, complexity: int) -> None:
        if self.unique_discovered >= self.limits.maximum_expressions:
            self.cap_rejected_unique += 1
            raise _CapReached("expression_cap_reached")
        if self.discovered_nodes + complexity > self.limits.maximum_total_nodes:
            self.cap_rejected_unique += 1
            raise _CapReached("node_cap_reached")
        self.unique_discovered += 1
        self.discovered_nodes += complexity

    def note_duplicate(self) -> None:
        self.duplicates_observed += 1

    def counts(self, *, emitted: bool) -> GrammarCounts:
        return GrammarCounts(
            self.work_units,
            self.unique_discovered,
            self.duplicates_observed,
            self.cap_rejected_unique,
            self.failed_work_units,
            self.unique_discovered if emitted else 0,
            self.discovered_nodes if emitted else 0,
        )


def _apply_unary(operator: str, value: sp.Expr) -> sp.Expr:
    if operator == "sqrt1p_minus1":
        return sp.sqrt(1 + value) - 1
    if operator == "saturate":
        return value / (1 + value)
    raise AssertionError("operator registry was not checked")


def _apply_binary(operator: str, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    if operator == "add":
        return left + right
    if operator == "multiply":
        return left * right
    raise AssertionError("operator registry was not checked")


def _enumerate(
    spec: GrammarSpec,
    counter: _Counter,
) -> tuple[_Generated, ...]:
    by_cost: dict[int, dict[str, _Generated]] = {}
    unique: dict[str, _Generated] = {}

    atom_bucket: dict[str, _Generated] = {}
    for atom in spec.atoms:
        counter.before_attempt()
        expression, key = canonicalize(SYMBOLS[atom])
        generated = _Generated(expression, 1, key, "atom", atom, ())
        atom_bucket[key] = generated
        if key not in unique:
            counter.before_unique(1)
            unique[key] = generated
        else:
            counter.note_duplicate()
    by_cost[1] = atom_bucket

    for cost in range(2, spec.maximum_complexity + 1):
        bucket: dict[str, _Generated] = {}
        for operator in spec.unary_operators:
            for child in by_cost.get(cost - 1, {}).values():
                counter.before_attempt()
                expression, key = canonicalize(_apply_unary(operator, child.expression))
                generated = _Generated(
                    expression,
                    cost,
                    key,
                    "unary",
                    operator,
                    (child.canonical,),
                )
                bucket.setdefault(key, generated)
                if key not in unique:
                    counter.before_unique(cost)
                    unique[key] = generated
                else:
                    counter.note_duplicate()

        for left_cost in range(1, cost - 1):
            right_cost = cost - 1 - left_cost
            pairs = product(
                by_cost.get(left_cost, {}).values(),
                by_cost.get(right_cost, {}).values(),
            )
            for left, right in pairs:
                for operator in spec.binary_operators:
                    counter.before_attempt()
                    expression, key = canonicalize(
                        _apply_binary(operator, left.expression, right.expression)
                    )
                    generated = _Generated(
                        expression,
                        cost,
                        key,
                        "binary",
                        operator,
                        (left.canonical, right.canonical),
                    )
                    bucket.setdefault(key, generated)
                    if key not in unique:
                        counter.before_unique(cost)
                        unique[key] = generated
                    else:
                        counter.note_duplicate()
        by_cost[cost] = bucket

    return tuple(sorted(unique.values(), key=lambda item: (item.complexity, item.canonical)))


def _manifest(
    *,
    spec: GrammarSpec,
    limits: GrammarLimits,
    domain_pack: DomainPackRef,
    sources: tuple[SourceBinding, ...],
    status: OutcomeStatus,
    reason_codes: Sequence[str],
    counts: GrammarCounts,
    candidates: Sequence[CandidateArtifact] = (),
    lineage: Sequence[GrammarLineage] = (),
) -> GrammarGenerationManifest:
    reasons = tuple(sorted(reason_codes))
    body = {
        "schema_version": SCHEMA_VERSION,
        "spec": spec.to_dict(),
        "limits": limits.to_dict(),
        "domain_pack": domain_pack.to_dict(),
        "sources": [item.to_dict() for item in sources],
        "status": status.value,
        "reason_codes": list(reasons),
        "counts": counts.to_dict(),
        "candidates": [item.to_dict() for item in candidates],
        "lineage": [item.to_dict() for item in lineage],
        "generation_only": True,
        "truth_established": False,
        "promotion_allowed": False,
    }
    return GrammarGenerationManifest(
        spec,
        limits,
        domain_pack,
        sources,
        status,
        reasons,
        counts,
        tuple(candidates),
        tuple(lineage),
        canonical_sha256(body),
    )


def _checked_sources(sources: Sequence[SourceBinding]) -> tuple[SourceBinding, ...]:
    ordered = tuple(sorted(sources, key=lambda item: item.role))
    if len(ordered) != len(_SOURCE_PATHS) or {item.role for item in ordered} != set(_SOURCE_PATHS):
        raise SchemaViolation("source bindings must contain exactly adapter and legacy_grammar")
    for item in ordered:
        if item.path != _SOURCE_PATHS[item.role]:
            raise SchemaViolation(f"source binding path changed for {item.role}")
    return ordered


def grammar_source_bindings(project_root: str | Path) -> tuple[SourceBinding, ...]:
    """Hash the exact adapter and legacy implementation files below ``project_root``."""

    root = Path(project_root).resolve()
    bindings = []
    for role, relative in sorted(_SOURCE_PATHS.items()):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SchemaViolation("grammar source escaped project root") from error
        if not path.is_file():
            raise SchemaViolation(f"grammar source is not a file: {relative}")
        bindings.append(
            SourceBinding(role, relative, hashlib.sha256(path.read_bytes()).hexdigest())
        )
    return tuple(bindings)


def generate_grammar_candidates(
    spec: GrammarSpec,
    limits: GrammarLimits,
    domain_pack: DomainPackRef,
    sources: Sequence[SourceBinding],
) -> GrammarGenerationManifest:
    """Enumerate a closed grammar and emit all-or-nothing Sigma Core artifacts."""

    if not isinstance(spec, GrammarSpec) or not isinstance(limits, GrammarLimits):
        raise SchemaViolation("grammar generation requires typed spec and limits")
    if not isinstance(domain_pack, DomainPackRef):
        raise SchemaViolation("grammar generation requires a DomainPackRef")
    checked_sources = _checked_sources(sources)
    reasons = []
    if set(spec.atoms) - set(REGISTERED_ATOMS):
        reasons.append("unknown_atom")
    if set(spec.unary_operators) - set(REGISTERED_UNARY_OPERATORS):
        reasons.append("unknown_unary_operator")
    if set(spec.binary_operators) - set(REGISTERED_BINARY_OPERATORS):
        reasons.append("unknown_binary_operator")
    if reasons:
        return _manifest(
            spec=spec,
            limits=limits,
            domain_pack=domain_pack,
            sources=checked_sources,
            status=OutcomeStatus.REJECT,
            reason_codes=reasons,
            counts=GrammarCounts(0, 0, 0, 0, 0, 0, 0),
        )

    counter = _Counter(limits)
    try:
        generated = _enumerate(spec, counter)
    except _CapReached as error:
        return _manifest(
            spec=spec,
            limits=limits,
            domain_pack=domain_pack,
            sources=checked_sources,
            status=OutcomeStatus.BLOCK,
            reason_codes=(error.reason_code,),
            counts=counter.counts(emitted=False),
        )
    # SymPy's canonicalizer exposes backend-specific exception types.  This is the deliberate
    # untrusted symbolic boundary: all ordinary failures become a typed, artifact-free error.
    except Exception:  # noqa: BLE001
        counter.failed_work_units += 1
        return _manifest(
            spec=spec,
            limits=limits,
            domain_pack=domain_pack,
            sources=checked_sources,
            status=OutcomeStatus.ERROR,
            reason_codes=("canonicalization_error",),
            counts=counter.counts(emitted=False),
        )

    parameters_base = {
        "adapter_schema_version": SCHEMA_VERSION,
        "spec": spec.to_dict(),
        "limits": limits.to_dict(),
        "generation_only": True,
        "truth_established": False,
        "promotion_allowed": False,
    }
    candidates = []
    by_canonical: dict[str, CandidateArtifact] = {}
    for ordinal, item in enumerate(generated):
        parent_inputs = {
            by_canonical[key].artifact_id: by_canonical[key].ref for key in item.parent_canonicals
        }
        parameters = {
            **parameters_base,
            "ordinal": ordinal,
            "canonical": item.canonical,
            "complexity": item.complexity,
            "derivation": {
                "operation": item.operation,
                "operator": item.operator,
                "parent_canonicals": list(item.parent_canonicals),
            },
        }
        candidate = CandidateArtifact.create(
            ArtifactKind.FORMULA,
            "A bounded closed-grammar expression was generated syntactically.",
            {
                "canonical_sympy": item.canonical,
                "expression": sp.sstr(item.expression),
                "complexity": item.complexity,
                "ordinal": ordinal,
                "grammar_spec_sha256": canonical_sha256(spec.to_dict()),
            },
            ProvenanceRecord.create(
                domain_pack,
                parameters,
                inputs=tuple(parent_inputs.values()),
                sources=checked_sources,
            ),
            assumptions=(
                "Generation uses only the declared closed grammar and exact node-cost bound.",
                "SymPy canonical equality is an algebraic deduplication key, not a proof of truth.",
            ),
            claims=("bounded_grammar_candidate",),
        )
        candidates.append(candidate)
        by_canonical[item.canonical] = candidate

    lineage = []
    for ordinal, (item, candidate) in enumerate(zip(generated, candidates, strict=True)):
        parents = [by_canonical[key].ref for key in item.parent_canonicals]
        lineage.append(
            GrammarLineage.create(
                ordinal=ordinal,
                operation=item.operation,
                operator=item.operator,
                child=candidate.ref,
                parents=parents,
                complexity=item.complexity,
            )
        )
    return _manifest(
        spec=spec,
        limits=limits,
        domain_pack=domain_pack,
        sources=checked_sources,
        status=OutcomeStatus.PASS,
        reason_codes=(),
        counts=counter.counts(emitted=True),
        candidates=candidates,
        lineage=lineage,
    )


def validate_grammar_manifest(
    value: GrammarGenerationManifest | Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> GrammarGenerationManifest:
    """Parse, re-enumerate, and byte-compare a manifest; optionally rehash source files."""

    parsed = (
        value
        if isinstance(value, GrammarGenerationManifest)
        else GrammarGenerationManifest.from_dict(value)
    )
    # Reparse even typed instances so mutable nested mappings cannot evade validation.
    parsed = GrammarGenerationManifest.from_dict(parsed.to_dict())
    if project_root is not None and parsed.sources != grammar_source_bindings(project_root):
        raise SchemaViolation("grammar manifest source bytes changed")
    replayed = generate_grammar_candidates(
        parsed.spec,
        parsed.limits,
        parsed.domain_pack,
        parsed.sources,
    )
    if parsed.to_dict() != replayed.to_dict():
        raise SchemaViolation("grammar manifest does not match deterministic replay")
    return parsed


__all__ = [
    "HARD_MAXIMUM_COMPLEXITY",
    "HARD_MAXIMUM_EXPRESSIONS",
    "HARD_MAXIMUM_TOTAL_NODES",
    "HARD_MAXIMUM_WORK_UNITS",
    "REGISTERED_ATOMS",
    "REGISTERED_BINARY_OPERATORS",
    "REGISTERED_UNARY_OPERATORS",
    "GrammarCounts",
    "GrammarGenerationManifest",
    "GrammarLimits",
    "GrammarLineage",
    "GrammarSpec",
    "generate_grammar_candidates",
    "grammar_source_bindings",
    "validate_grammar_manifest",
]
