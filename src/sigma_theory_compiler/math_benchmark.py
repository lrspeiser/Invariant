"""Closed benchmark manifests, dependency isolation, and proof-first scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .sigma_core import canonical_sha256

_ID = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class BenchmarkBoundaryError(ValueError):
    """Raised when a benchmark or dependency closure is not fail-closed."""


class MathObjectKind(str, Enum):
    OBJECT = "object"
    DEFINITION = "definition"
    AXIOM = "axiom"
    THEOREM = "theorem"
    CONJECTURE = "conjecture"
    LEMMA = "lemma"
    IDENTITY = "identity"
    CONSTRUCTION = "construction"
    ALGORITHM = "algorithm"
    COUNTEREXAMPLE = "counterexample"
    PROOF = "proof"
    DERIVATION = "derivation"


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise BenchmarkBoundaryError(f"{label} is not a canonical identifier")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise BenchmarkBoundaryError(f"{label} is not a lowercase SHA-256")
    return value


def _sorted_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    checked = tuple(_identifier(value, label) for value in values)
    if checked != tuple(sorted(set(checked))):
        raise BenchmarkBoundaryError(f"{label} values must be sorted and unique")
    return checked


@dataclass(frozen=True, slots=True)
class KnowledgeNode:
    node_id: str
    kind: MathObjectKind
    content_sha256: str
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.node_id, "knowledge node ID")
        if not isinstance(self.kind, MathObjectKind):
            raise BenchmarkBoundaryError("knowledge node kind is unregistered")
        _hash(self.content_sha256, "knowledge node content hash")
        _sorted_unique(self.dependencies, "dependency")
        if self.node_id in self.dependencies:
            raise BenchmarkBoundaryError("knowledge node cannot depend on itself")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "content_sha256": self.content_sha256,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True, slots=True)
class MathematicalKnowledgeGraph:
    nodes: tuple[KnowledgeNode, ...]
    schema_version: str = "sigma-math-knowledge-graph-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-math-knowledge-graph-1.0":
            raise BenchmarkBoundaryError("knowledge graph schema changed")
        if not self.nodes or self.nodes != tuple(sorted(self.nodes, key=lambda node: node.node_id)):
            raise BenchmarkBoundaryError("knowledge nodes must be nonempty and sorted")
        node_map = {node.node_id: node for node in self.nodes}
        if len(node_map) != len(self.nodes):
            raise BenchmarkBoundaryError("knowledge node IDs contain duplicates")
        for node in self.nodes:
            if not set(node.dependencies) <= set(node_map):
                raise BenchmarkBoundaryError("knowledge dependency is unregistered")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise BenchmarkBoundaryError("knowledge dependency graph contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in node_map[node_id].dependencies:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in node_map:
            visit(node_id)

    @property
    def graph_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def node(self, node_id: str) -> KnowledgeNode:
        try:
            return next(node for node in self.nodes if node.node_id == node_id)
        except StopIteration as error:
            raise BenchmarkBoundaryError(f"unknown knowledge node: {node_id}") from error

    def dependency_closure(self, node_ids: tuple[str, ...]) -> tuple[str, ...]:
        pending = list(_sorted_unique(node_ids, "closure seed"))
        closed: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in closed:
                continue
            node = self.node(node_id)
            closed.add(node_id)
            pending.extend(node.dependencies)
        return tuple(sorted(closed))

    def descendants(self, node_id: str) -> tuple[str, ...]:
        self.node(node_id)
        descendants = {
            candidate.node_id
            for candidate in self.nodes
            if node_id in self.dependency_closure((candidate.node_id,))
        }
        return tuple(sorted(descendants))


@dataclass(frozen=True, slots=True)
class BlindHoldoutManifest:
    benchmark_id: str
    target_node_id: str
    knowledge_graph_sha256: str
    generation_visible: tuple[str, ...]
    verification_allowed: tuple[str, ...]
    pre_unseal_forbidden: tuple[str, ...]
    post_unseal_visible: tuple[str, ...]
    schema_version: str = "sigma-math-blind-holdout-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-math-blind-holdout-1.0":
            raise BenchmarkBoundaryError("blind holdout schema changed")
        _identifier(self.benchmark_id, "benchmark ID")
        _identifier(self.target_node_id, "target node ID")
        _hash(self.knowledge_graph_sha256, "knowledge graph hash")
        for label in (
            "generation_visible",
            "verification_allowed",
            "pre_unseal_forbidden",
            "post_unseal_visible",
        ):
            _sorted_unique(getattr(self, label), label)
        if (
            self.target_node_id in self.generation_visible
            or self.target_node_id in self.verification_allowed
        ):
            raise BenchmarkBoundaryError("target is exposed before unseal")
        if self.target_node_id not in self.pre_unseal_forbidden:
            raise BenchmarkBoundaryError("target must be explicitly forbidden before unseal")
        if self.target_node_id not in self.post_unseal_visible:
            raise BenchmarkBoundaryError("target must be available for post-unseal comparison")
        if (set(self.generation_visible) | set(self.verification_allowed)) & set(
            self.pre_unseal_forbidden
        ):
            raise BenchmarkBoundaryError("allowed and forbidden pre-unseal closures overlap")

    def validate_against(self, graph: MathematicalKnowledgeGraph) -> None:
        if graph.graph_sha256 != self.knowledge_graph_sha256:
            raise BenchmarkBoundaryError("manifest knowledge graph binding changed")
        all_ids = {
            self.target_node_id,
            *self.generation_visible,
            *self.verification_allowed,
            *self.pre_unseal_forbidden,
            *self.post_unseal_visible,
        }
        for node_id in all_ids:
            graph.node(node_id)
        ancestors = set(graph.dependency_closure((self.target_node_id,))) - {self.target_node_id}
        if not set(self.generation_visible) <= ancestors:
            raise BenchmarkBoundaryError("generation closure contains a non-ancestor of the target")
        descendants = set(graph.descendants(self.target_node_id))
        if not descendants <= set(self.pre_unseal_forbidden):
            raise BenchmarkBoundaryError("target-dependent downstream knowledge is not forbidden")

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "target_node_id": self.target_node_id,
            "knowledge_graph_sha256": self.knowledge_graph_sha256,
            "generation_visible": list(self.generation_visible),
            "verification_allowed": list(self.verification_allowed),
            "pre_unseal_forbidden": list(self.pre_unseal_forbidden),
            "post_unseal_visible": list(self.post_unseal_visible),
        }

    def check_proof_dependencies(
        self,
        graph: MathematicalKnowledgeGraph,
        dependencies: tuple[str, ...],
    ) -> tuple[str, ...]:
        self.validate_against(graph)
        declared = set(_sorted_unique(dependencies, "proof dependency"))
        closure = set(graph.dependency_closure(tuple(sorted(declared)))) if declared else set()
        allowed = set(self.generation_visible) | set(self.verification_allowed)
        forbidden = closure & set(self.pre_unseal_forbidden)
        undeclared = closure - allowed
        if forbidden or undeclared:
            raise BenchmarkBoundaryError(
                "proof dependency closure escapes the allowed premise manifest"
            )
        return tuple(sorted(closure))


@dataclass(frozen=True, slots=True)
class DiscoveryScore:
    well_typed: bool
    nontrivial: bool
    survived_counterexamples: bool
    computational_domain_passed: bool
    formally_proved: bool
    forbidden_premises_absent: bool
    post_unseal_comparison_complete: bool
    simplicity_cost: int

    def __post_init__(self) -> None:
        flags = (
            self.well_typed,
            self.nontrivial,
            self.survived_counterexamples,
            self.computational_domain_passed,
            self.formally_proved,
            self.forbidden_premises_absent,
            self.post_unseal_comparison_complete,
        )
        if any(type(flag) is not bool for flag in flags):
            raise BenchmarkBoundaryError("discovery hard gates must be booleans")
        if (
            isinstance(self.simplicity_cost, bool)
            or not isinstance(self.simplicity_cost, int)
            or self.simplicity_cost < 0
        ):
            raise BenchmarkBoundaryError("simplicity cost must be a nonnegative integer")

    @property
    def discovered_and_proved(self) -> bool:
        return all(
            (
                self.well_typed,
                self.nontrivial,
                self.survived_counterexamples,
                self.computational_domain_passed,
                self.formally_proved,
                self.forbidden_premises_absent,
                self.post_unseal_comparison_complete,
            )
        )

    @property
    def lexicographic_key(self) -> tuple[int, ...]:
        return (
            int(self.well_typed),
            int(self.nontrivial),
            int(self.survived_counterexamples),
            int(self.computational_domain_passed),
            int(self.formally_proved),
            int(self.forbidden_premises_absent),
            int(self.post_unseal_comparison_complete),
            -self.simplicity_cost,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkTally:
    eligible_holdouts: int
    rediscovered: int
    formally_proved: int
    dependency_clean: int
    leakage_failures: int
    false_positives: int

    def __post_init__(self) -> None:
        values = (
            self.eligible_holdouts,
            self.rediscovered,
            self.formally_proved,
            self.dependency_clean,
            self.leakage_failures,
            self.false_positives,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise BenchmarkBoundaryError("benchmark counts must be nonnegative integers")
        if not self.formally_proved <= self.rediscovered <= self.eligible_holdouts:
            raise BenchmarkBoundaryError("proved/rediscovered/eligible counts are inconsistent")
        if self.dependency_clean > self.formally_proved:
            raise BenchmarkBoundaryError("dependency-clean proofs exceed formal proofs")
        if self.leakage_failures + self.dependency_clean > self.eligible_holdouts:
            raise BenchmarkBoundaryError("leakage and dependency-clean counts exceed holdouts")

    @property
    def proved_rate(self) -> dict[str, int]:
        return {"numerator": self.dependency_clean, "denominator": self.eligible_holdouts}


__all__ = [
    "BenchmarkBoundaryError",
    "BenchmarkTally",
    "BlindHoldoutManifest",
    "DiscoveryScore",
    "KnowledgeNode",
    "MathObjectKind",
    "MathematicalKnowledgeGraph",
]
