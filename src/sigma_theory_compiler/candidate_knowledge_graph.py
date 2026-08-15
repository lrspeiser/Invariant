"""Closed, domain-independent knowledge graph for Sigma candidate artifacts.

Corpus lookup can establish presence or absence from this exact sealed corpus only.  It never
establishes novelty, truth, proof, scientific validity, or promotion eligibility.
"""

from __future__ import annotations

import heapq
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sigma_theory_compiler.sigma_core import (
    ArtifactRef,
    CandidateArtifact,
    SchemaViolation,
    SourceBinding,
    canonical_json_bytes,
    canonical_sha256,
)

GRAPH_SCHEMA = "sigma-candidate-knowledge-graph-1.0"
NODE_SCHEMA = "sigma-candidate-knowledge-node-1.0"
EDGE_SCHEMA = "sigma-candidate-knowledge-edge-1.0"
HOLDOUT_SCHEMA = "sigma-candidate-knowledge-holdout-1.0"
SCOPE = (
    "Exact sealed corpus membership and typed graph relations only; absence means absent from "
    "this corpus and never establishes novelty, truth, proof, validity, or promotion."
)
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARTIFACTS = 10_000
_MAX_NODES = 50_000
_MAX_EDGES = 200_000
_MAX_DEPENDENCY_DEPTH = 10_000
_MAX_EQUIVALENCE_CLASS_SIZE = 10_000


class CandidateKnowledgeGraphError(ValueError):
    """A graph value or query violates the closed knowledge boundary."""


class KnowledgeNodeKind(str, Enum):
    ARTIFACT = "artifact"
    DEFINITION = "definition"
    AXIOM = "axiom"
    LEMMA = "lemma"
    THEOREM = "theorem"
    CONJECTURE = "conjecture"
    IDENTITY = "identity"
    CONSTRUCTION = "construction"
    ALGORITHM = "algorithm"
    COUNTEREXAMPLE = "counterexample"
    PROOF = "proof"
    DERIVATION = "derivation"


class KnowledgeEdgeKind(str, Enum):
    DEPENDENCY = "dependency"
    EQUIVALENCE = "equivalence"
    PROVES = "proves"
    DERIVES = "derives"


class CorpusPresence(str, Enum):
    PRESENT = "present_in_this_corpus"
    ABSENT = "absent_from_this_corpus"


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CandidateKnowledgeGraphError(f"{label} keys changed")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise CandidateKnowledgeGraphError(f"{label} is not a canonical identifier")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CandidateKnowledgeGraphError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _bounded_positive(value: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise CandidateKnowledgeGraphError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _canonical_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateKnowledgeGraphError(f"{label} must be an object")
    try:
        detached = json.loads(canonical_json_bytes(value))
    except (SchemaViolation, TypeError, ValueError) as error:
        raise CandidateKnowledgeGraphError(f"{label} is not canonical exact JSON") from error
    if not isinstance(detached, dict):
        raise CandidateKnowledgeGraphError(f"{label} must be an object")
    return detached


@dataclass(frozen=True, slots=True)
class KnowledgeGraphLimits:
    max_artifacts: int = _MAX_ARTIFACTS
    max_nodes: int = _MAX_NODES
    max_edges: int = _MAX_EDGES
    max_dependency_depth: int = _MAX_DEPENDENCY_DEPTH
    max_equivalence_class_size: int = _MAX_EQUIVALENCE_CLASS_SIZE

    def __post_init__(self) -> None:
        _bounded_positive(self.max_artifacts, _MAX_ARTIFACTS, "max_artifacts")
        _bounded_positive(self.max_nodes, _MAX_NODES, "max_nodes")
        _bounded_positive(self.max_edges, _MAX_EDGES, "max_edges")
        _bounded_positive(self.max_dependency_depth, _MAX_DEPENDENCY_DEPTH, "max_dependency_depth")
        _bounded_positive(
            self.max_equivalence_class_size,
            _MAX_EQUIVALENCE_CLASS_SIZE,
            "max_equivalence_class_size",
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_artifacts": self.max_artifacts,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_dependency_depth": self.max_dependency_depth,
            "max_equivalence_class_size": self.max_equivalence_class_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KnowledgeGraphLimits:
        _exact_keys(
            value,
            {
                "max_artifacts",
                "max_nodes",
                "max_edges",
                "max_dependency_depth",
                "max_equivalence_class_size",
            },
            "knowledge graph limits",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class CandidateKnowledgeNode:
    node_id: str
    kind: KnowledgeNodeKind
    payload: Mapping[str, Any]
    artifact_ref: ArtifactRef | None
    sources: tuple[SourceBinding, ...]
    content_sha256: str
    schema_version: str = NODE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != NODE_SCHEMA:
            raise CandidateKnowledgeGraphError("knowledge node schema_version changed")
        if not isinstance(self.kind, KnowledgeNodeKind):
            raise CandidateKnowledgeGraphError("knowledge node kind is unregistered")
        payload = _canonical_object(self.payload, "knowledge node payload")
        object.__setattr__(self, "payload", payload)
        if self.artifact_ref is not None and not isinstance(self.artifact_ref, ArtifactRef):
            raise CandidateKnowledgeGraphError("node artifact_ref must be a Sigma ArtifactRef")
        if any(not isinstance(item, SourceBinding) for item in self.sources):
            raise CandidateKnowledgeGraphError("node sources must be Sigma SourceBinding values")
        if self.kind is KnowledgeNodeKind.ARTIFACT and self.artifact_ref is None:
            raise CandidateKnowledgeGraphError("artifact node requires an ArtifactRef")
        roles = tuple(item.role for item in self.sources)
        if roles != tuple(sorted(set(roles))):
            raise CandidateKnowledgeGraphError("node source roles must be unique and sorted")
        _hash(self.content_sha256, "knowledge node content_sha256")
        expected = canonical_sha256(self._body())
        if self.content_sha256 != expected or self.node_id != f"ckn-{expected[:24]}":
            raise CandidateKnowledgeGraphError("knowledge node canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "payload": self.payload,
            "artifact_ref": None if self.artifact_ref is None else self.artifact_ref.to_dict(),
            "sources": [item.to_dict() for item in self.sources],
        }

    @classmethod
    def create(
        cls,
        kind: KnowledgeNodeKind,
        payload: Mapping[str, Any],
        *,
        artifact_ref: ArtifactRef | None = None,
        sources: Sequence[SourceBinding] = (),
    ) -> CandidateKnowledgeNode:
        if not isinstance(kind, KnowledgeNodeKind):
            raise CandidateKnowledgeGraphError("knowledge node kind is unregistered")
        if artifact_ref is not None and not isinstance(artifact_ref, ArtifactRef):
            raise CandidateKnowledgeGraphError("node artifact_ref must be a Sigma ArtifactRef")
        if any(not isinstance(item, SourceBinding) for item in sources):
            raise CandidateKnowledgeGraphError("node sources must be Sigma SourceBinding values")
        ordered_sources = tuple(sorted(sources, key=lambda item: item.role))
        body = {
            "schema_version": NODE_SCHEMA,
            "kind": kind.value,
            "payload": _canonical_object(payload, "knowledge node payload"),
            "artifact_ref": None if artifact_ref is None else artifact_ref.to_dict(),
            "sources": [item.to_dict() for item in ordered_sources],
        }
        digest = canonical_sha256(body)
        return cls(
            node_id=f"ckn-{digest[:24]}",
            kind=kind,
            payload=body["payload"],
            artifact_ref=artifact_ref,
            sources=ordered_sources,
            content_sha256=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, **self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateKnowledgeNode:
        _exact_keys(
            value,
            {
                "node_id",
                "schema_version",
                "kind",
                "payload",
                "artifact_ref",
                "sources",
                "content_sha256",
            },
            "knowledge node",
        )
        if not isinstance(value["sources"], list):
            raise CandidateKnowledgeGraphError("knowledge node sources must be an array")
        try:
            kind = KnowledgeNodeKind(value["kind"])
            artifact_ref = (
                None
                if value["artifact_ref"] is None
                else ArtifactRef.from_dict(value["artifact_ref"])
            )
            sources = tuple(SourceBinding.from_dict(item) for item in value["sources"])
        except (SchemaViolation, TypeError, ValueError) as error:
            raise CandidateKnowledgeGraphError(
                "knowledge node Sigma Core binding failed"
            ) from error
        return cls(
            node_id=str(value["node_id"]),
            kind=kind,
            payload=value["payload"],
            artifact_ref=artifact_ref,
            sources=sources,
            content_sha256=str(value["content_sha256"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class CandidateKnowledgeEdge:
    edge_id: str
    kind: KnowledgeEdgeKind
    source_node_id: str
    target_node_id: str
    content_sha256: str
    schema_version: str = EDGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EDGE_SCHEMA:
            raise CandidateKnowledgeGraphError("knowledge edge schema_version changed")
        if not isinstance(self.kind, KnowledgeEdgeKind):
            raise CandidateKnowledgeGraphError("knowledge edge kind is unregistered")
        _identifier(self.source_node_id, "edge source node ID")
        _identifier(self.target_node_id, "edge target node ID")
        if self.source_node_id == self.target_node_id:
            raise CandidateKnowledgeGraphError("knowledge edge cannot be a self-loop")
        if self.kind is KnowledgeEdgeKind.EQUIVALENCE and not (
            self.source_node_id < self.target_node_id
        ):
            raise CandidateKnowledgeGraphError("equivalence edge endpoints must be canonicalized")
        _hash(self.content_sha256, "knowledge edge content_sha256")
        expected = canonical_sha256(self._body())
        if self.content_sha256 != expected or self.edge_id != f"cke-{expected[:24]}":
            raise CandidateKnowledgeGraphError("knowledge edge canonical identity changed")

    def _body(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
        }

    @classmethod
    def create(
        cls, kind: KnowledgeEdgeKind, source_node_id: str, target_node_id: str
    ) -> CandidateKnowledgeEdge:
        if not isinstance(kind, KnowledgeEdgeKind):
            raise CandidateKnowledgeGraphError("knowledge edge kind is unregistered")
        if kind is KnowledgeEdgeKind.EQUIVALENCE and target_node_id < source_node_id:
            source_node_id, target_node_id = target_node_id, source_node_id
        body = {
            "schema_version": EDGE_SCHEMA,
            "kind": kind.value,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        digest = canonical_sha256(body)
        return cls(
            edge_id=f"cke-{digest[:24]}",
            kind=kind,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            content_sha256=digest,
        )

    def to_dict(self) -> dict[str, str]:
        return {"edge_id": self.edge_id, **self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateKnowledgeEdge:
        _exact_keys(
            value,
            {
                "edge_id",
                "schema_version",
                "kind",
                "source_node_id",
                "target_node_id",
                "content_sha256",
            },
            "knowledge edge",
        )
        try:
            kind = KnowledgeEdgeKind(value["kind"])
        except (TypeError, ValueError) as error:
            raise CandidateKnowledgeGraphError("knowledge edge kind is unregistered") from error
        return cls(
            edge_id=str(value["edge_id"]),
            kind=kind,
            source_node_id=str(value["source_node_id"]),
            target_node_id=str(value["target_node_id"]),
            content_sha256=str(value["content_sha256"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class CorpusLookup:
    queried_content_sha256: str
    status: CorpusPresence
    node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _hash(self.queried_content_sha256, "queried content_sha256")
        if self.node_ids != tuple(sorted(set(self.node_ids))):
            raise CandidateKnowledgeGraphError("corpus lookup node IDs must be unique and sorted")
        expected = CorpusPresence.PRESENT if self.node_ids else CorpusPresence.ABSENT
        if self.status is not expected:
            raise CandidateKnowledgeGraphError("corpus lookup status changed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "queried_content_sha256": self.queried_content_sha256,
            "status": self.status.value,
            "node_ids": list(self.node_ids),
            "scope": "absence is limited to this exact corpus and is not a novelty claim",
        }


@dataclass(frozen=True, slots=True)
class HoldoutCut:
    graph_sha256: str
    target_node_id: str
    target_equivalence_class: tuple[str, ...]
    forbidden_node_ids: tuple[str, ...]
    visible_node_ids: tuple[str, ...]
    content_sha256: str
    schema_version: str = HOLDOUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != HOLDOUT_SCHEMA:
            raise CandidateKnowledgeGraphError("holdout schema_version changed")
        _hash(self.graph_sha256, "holdout graph_sha256")
        _identifier(self.target_node_id, "holdout target_node_id")
        for label in ("target_equivalence_class", "forbidden_node_ids", "visible_node_ids"):
            values = getattr(self, label)
            if values != tuple(sorted(set(values))):
                raise CandidateKnowledgeGraphError(f"holdout {label} must be unique and sorted")
        if self.target_node_id not in self.target_equivalence_class:
            raise CandidateKnowledgeGraphError(
                "holdout target is absent from its equivalence class"
            )
        if not set(self.target_equivalence_class) <= set(self.forbidden_node_ids):
            raise CandidateKnowledgeGraphError("holdout target class is not fully forbidden")
        if set(self.forbidden_node_ids) & set(self.visible_node_ids):
            raise CandidateKnowledgeGraphError("holdout visible and forbidden sets overlap")
        _hash(self.content_sha256, "holdout content_sha256")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise CandidateKnowledgeGraphError("holdout canonical hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_sha256": self.graph_sha256,
            "target_node_id": self.target_node_id,
            "target_equivalence_class": list(self.target_equivalence_class),
            "forbidden_node_ids": list(self.forbidden_node_ids),
            "visible_node_ids": list(self.visible_node_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    def validate_against(self, graph: CandidateKnowledgeGraph) -> None:
        if not isinstance(graph, CandidateKnowledgeGraph):
            raise CandidateKnowledgeGraphError("holdout validation requires a knowledge graph")
        if self.graph_sha256 != graph.content_sha256:
            raise CandidateKnowledgeGraphError("holdout graph binding changed")
        if self != graph.holdout_cut(self.target_node_id):
            raise CandidateKnowledgeGraphError("holdout cut is not the complete target downstream")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HoldoutCut:
        _exact_keys(
            value,
            {
                "schema_version",
                "graph_sha256",
                "target_node_id",
                "target_equivalence_class",
                "forbidden_node_ids",
                "visible_node_ids",
                "content_sha256",
            },
            "holdout cut",
        )
        for key in ("target_equivalence_class", "forbidden_node_ids", "visible_node_ids"):
            if not isinstance(value[key], list):
                raise CandidateKnowledgeGraphError(f"holdout {key} must be an array")
        return cls(
            graph_sha256=str(value["graph_sha256"]),
            target_node_id=str(value["target_node_id"]),
            target_equivalence_class=tuple(value["target_equivalence_class"]),
            forbidden_node_ids=tuple(value["forbidden_node_ids"]),
            visible_node_ids=tuple(value["visible_node_ids"]),
            content_sha256=str(value["content_sha256"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class CandidateKnowledgeGraph:
    corpus_id: str
    artifacts: tuple[CandidateArtifact, ...]
    nodes: tuple[CandidateKnowledgeNode, ...]
    edges: tuple[CandidateKnowledgeEdge, ...]
    limits: KnowledgeGraphLimits
    content_sha256: str
    scope: str = SCOPE
    schema_version: str = GRAPH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GRAPH_SCHEMA or self.scope != SCOPE:
            raise CandidateKnowledgeGraphError("knowledge graph schema or scope changed")
        _identifier(self.corpus_id, "corpus_id")
        if not self.nodes:
            raise CandidateKnowledgeGraphError("knowledge graph must contain at least one node")
        if any(not isinstance(item, CandidateArtifact) for item in self.artifacts):
            raise CandidateKnowledgeGraphError("graph artifacts must be CandidateArtifact values")
        if any(not isinstance(item, CandidateKnowledgeNode) for item in self.nodes):
            raise CandidateKnowledgeGraphError("graph nodes must be CandidateKnowledgeNode values")
        if any(not isinstance(item, CandidateKnowledgeEdge) for item in self.edges):
            raise CandidateKnowledgeGraphError("graph edges must be CandidateKnowledgeEdge values")
        if len(self.artifacts) > self.limits.max_artifacts:
            raise CandidateKnowledgeGraphError("knowledge graph artifact limit exceeded")
        if len(self.nodes) > self.limits.max_nodes:
            raise CandidateKnowledgeGraphError("knowledge graph node limit exceeded")
        if len(self.edges) > self.limits.max_edges:
            raise CandidateKnowledgeGraphError("knowledge graph edge limit exceeded")
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        if artifact_ids != tuple(sorted(set(artifact_ids))):
            raise CandidateKnowledgeGraphError("graph artifacts must be unique and sorted")
        artifact_map = {item.artifact_id: item for item in self.artifacts}
        for artifact in self.artifacts:
            try:
                artifact.validate()
            except SchemaViolation as error:
                raise CandidateKnowledgeGraphError(
                    "graph candidate artifact failed validation"
                ) from error
        node_ids = tuple(item.node_id for item in self.nodes)
        if node_ids != tuple(sorted(set(node_ids))):
            raise CandidateKnowledgeGraphError("graph nodes must be unique and sorted")
        node_map = {item.node_id: item for item in self.nodes}
        edge_ids = tuple(item.edge_id for item in self.edges)
        if edge_ids != tuple(sorted(set(edge_ids))):
            raise CandidateKnowledgeGraphError("graph edges must be unique and sorted")
        for node in self.nodes:
            if node.artifact_ref is None:
                continue
            artifact = artifact_map.get(node.artifact_ref.artifact_id)
            if artifact is None or artifact.content_sha256 != node.artifact_ref.content_sha256:
                raise CandidateKnowledgeGraphError("node artifact reference is dangling or changed")
        referenced_artifacts = {
            node.artifact_ref.artifact_id for node in self.nodes if node.artifact_ref is not None
        }
        if referenced_artifacts != set(artifact_map):
            raise CandidateKnowledgeGraphError(
                "graph artifact registry contains unreferenced entries"
            )
        for edge in self.edges:
            if edge.source_node_id not in node_map or edge.target_node_id not in node_map:
                raise CandidateKnowledgeGraphError("knowledge edge has a dangling endpoint")
            source = node_map[edge.source_node_id]
            target = node_map[edge.target_node_id]
            if edge.kind is KnowledgeEdgeKind.PROVES and (
                source.kind is not KnowledgeNodeKind.PROOF
                or target.kind
                not in {
                    KnowledgeNodeKind.LEMMA,
                    KnowledgeNodeKind.THEOREM,
                    KnowledgeNodeKind.IDENTITY,
                }
            ):
                raise CandidateKnowledgeGraphError("proof edge has incompatible endpoint kinds")
            if edge.kind is KnowledgeEdgeKind.DERIVES and (
                source.kind is not KnowledgeNodeKind.DERIVATION
                or target.kind
                not in {
                    KnowledgeNodeKind.ARTIFACT,
                    KnowledgeNodeKind.LEMMA,
                    KnowledgeNodeKind.THEOREM,
                    KnowledgeNodeKind.IDENTITY,
                    KnowledgeNodeKind.CONSTRUCTION,
                    KnowledgeNodeKind.ALGORITHM,
                }
            ):
                raise CandidateKnowledgeGraphError(
                    "derivation edge has incompatible endpoint kinds"
                )
        proof_sources = {
            edge.source_node_id for edge in self.edges if edge.kind is KnowledgeEdgeKind.PROVES
        }
        derivation_sources = {
            edge.source_node_id for edge in self.edges if edge.kind is KnowledgeEdgeKind.DERIVES
        }
        if any(
            node.kind is KnowledgeNodeKind.PROOF and node.node_id not in proof_sources
            for node in self.nodes
        ):
            raise CandidateKnowledgeGraphError("proof node has no proof edge")
        if any(
            node.kind is KnowledgeNodeKind.DERIVATION and node.node_id not in derivation_sources
            for node in self.nodes
        ):
            raise CandidateKnowledgeGraphError("derivation node has no derivation edge")
        self._validate_dependency_dag()
        equivalence_components = self._equivalence_components()
        if any(
            len(component) > self.limits.max_equivalence_class_size
            for component in set(equivalence_components.values())
        ):
            raise CandidateKnowledgeGraphError("equivalence class size limit exceeded")
        _hash(self.content_sha256, "knowledge graph content_sha256")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise CandidateKnowledgeGraphError("knowledge graph canonical hash changed")

    def _dependency_map(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {item.node_id: [] for item in self.nodes}
        for edge in self.edges:
            if edge.kind is KnowledgeEdgeKind.DEPENDENCY:
                result[edge.source_node_id].append(edge.target_node_id)
        return {key: tuple(sorted(value)) for key, value in result.items()}

    def _validate_dependency_dag(self) -> None:
        dependencies = self._dependency_map()
        dependents: dict[str, list[str]] = {node_id: [] for node_id in dependencies}
        remaining = {node_id: len(targets) for node_id, targets in dependencies.items()}
        for node_id, targets in dependencies.items():
            for target in targets:
                dependents[target].append(node_id)
        ready = [node_id for node_id, count in remaining.items() if count == 0]
        heapq.heapify(ready)
        depths = {node_id: 1 for node_id in ready}
        processed = 0
        while ready:
            node_id = heapq.heappop(ready)
            processed += 1
            for dependent in sorted(dependents[node_id]):
                remaining[dependent] -= 1
                depths[dependent] = max(depths.get(dependent, 1), depths[node_id] + 1)
                if remaining[dependent] == 0:
                    heapq.heappush(ready, dependent)
        if processed != len(self.nodes):
            raise CandidateKnowledgeGraphError("dependency graph contains a cycle")
        if depths and max(depths.values()) > self.limits.max_dependency_depth:
            raise CandidateKnowledgeGraphError("dependency graph depth limit exceeded")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "corpus_id": self.corpus_id,
            "limits": self.limits.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
        }

    @classmethod
    def create(
        cls,
        corpus_id: str,
        *,
        artifacts: Sequence[CandidateArtifact],
        nodes: Sequence[CandidateKnowledgeNode],
        edges: Sequence[CandidateKnowledgeEdge],
        limits: KnowledgeGraphLimits | None = None,
    ) -> CandidateKnowledgeGraph:
        if any(not isinstance(item, CandidateArtifact) for item in artifacts):
            raise CandidateKnowledgeGraphError("graph artifacts must be CandidateArtifact values")
        if any(not isinstance(item, CandidateKnowledgeNode) for item in nodes):
            raise CandidateKnowledgeGraphError("graph nodes must be CandidateKnowledgeNode values")
        if any(not isinstance(item, CandidateKnowledgeEdge) for item in edges):
            raise CandidateKnowledgeGraphError("graph edges must be CandidateKnowledgeEdge values")
        limits = limits or KnowledgeGraphLimits()
        if not isinstance(limits, KnowledgeGraphLimits):
            raise CandidateKnowledgeGraphError("limits must be KnowledgeGraphLimits")
        ordered_artifacts = tuple(sorted(artifacts, key=lambda item: item.artifact_id))
        ordered_nodes = tuple(sorted(nodes, key=lambda item: item.node_id))
        ordered_edges = tuple(sorted(edges, key=lambda item: item.edge_id))
        provisional = object.__new__(cls)
        fields = {
            "corpus_id": corpus_id,
            "artifacts": ordered_artifacts,
            "nodes": ordered_nodes,
            "edges": ordered_edges,
            "limits": limits,
            "scope": SCOPE,
            "schema_version": GRAPH_SCHEMA,
        }
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        return cls(**fields, content_sha256=canonical_sha256(provisional._body()))

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateKnowledgeGraph:
        _exact_keys(
            value,
            {
                "schema_version",
                "scope",
                "corpus_id",
                "limits",
                "artifacts",
                "nodes",
                "edges",
                "content_sha256",
            },
            "candidate knowledge graph",
        )
        for key in ("artifacts", "nodes", "edges"):
            if not isinstance(value[key], list):
                raise CandidateKnowledgeGraphError(f"knowledge graph {key} must be an array")
        try:
            artifacts = tuple(CandidateArtifact.from_dict(item) for item in value["artifacts"])
        except (SchemaViolation, TypeError, ValueError) as error:
            raise CandidateKnowledgeGraphError(
                "graph candidate artifact failed validation"
            ) from error
        return cls(
            corpus_id=str(value["corpus_id"]),
            artifacts=artifacts,
            nodes=tuple(CandidateKnowledgeNode.from_dict(item) for item in value["nodes"]),
            edges=tuple(CandidateKnowledgeEdge.from_dict(item) for item in value["edges"]),
            limits=KnowledgeGraphLimits.from_dict(value["limits"]),
            content_sha256=str(value["content_sha256"]),
            scope=str(value["scope"]),
            schema_version=str(value["schema_version"]),
        )

    def node(self, node_id: str) -> CandidateKnowledgeNode:
        _identifier(node_id, "query node ID")
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise CandidateKnowledgeGraphError(f"unknown knowledge node: {node_id}")

    def nodes_by_kind(self, kind: KnowledgeNodeKind) -> tuple[CandidateKnowledgeNode, ...]:
        if not isinstance(kind, KnowledgeNodeKind):
            raise CandidateKnowledgeGraphError("query node kind is unregistered")
        return tuple(node for node in self.nodes if node.kind is kind)

    def direct_dependencies(self, node_id: str) -> tuple[str, ...]:
        self.node(node_id)
        return self._dependency_map()[node_id]

    def dependency_closure(self, node_ids: Sequence[str]) -> tuple[str, ...]:
        if not node_ids:
            return ()
        seeds = tuple(sorted(set(node_ids)))
        if len(seeds) != len(node_ids):
            raise CandidateKnowledgeGraphError("dependency closure seeds contain duplicates")
        dependencies = self._dependency_map()
        pending = list(seeds)
        closed: set[str] = set()
        while pending:
            node_id = pending.pop()
            self.node(node_id)
            if node_id in closed:
                continue
            closed.add(node_id)
            pending.extend(dependencies[node_id])
        return tuple(sorted(closed))

    def downstream(self, node_id: str) -> tuple[str, ...]:
        self.node(node_id)
        dependents: dict[str, set[str]] = {node.node_id: set() for node in self.nodes}
        for edge in self.edges:
            if edge.kind is KnowledgeEdgeKind.DEPENDENCY:
                dependents[edge.target_node_id].add(edge.source_node_id)
        pending = [node_id]
        closed: set[str] = set()
        while pending:
            current = pending.pop()
            if current in closed:
                continue
            closed.add(current)
            pending.extend(dependents[current] - closed)
        return tuple(sorted(closed))

    def _equivalence_components(self) -> dict[str, tuple[str, ...]]:
        parent = {node.node_id: node.node_id for node in self.nodes}

        def find(node_id: str) -> str:
            root = node_id
            while parent[root] != root:
                root = parent[root]
            while parent[node_id] != node_id:
                next_id = parent[node_id]
                parent[node_id] = root
                node_id = next_id
            return root

        for edge in self.edges:
            if edge.kind is not KnowledgeEdgeKind.EQUIVALENCE:
                continue
            left = find(edge.source_node_id)
            right = find(edge.target_node_id)
            if left != right:
                smaller, larger = sorted((left, right))
                parent[larger] = smaller
        grouped: dict[str, list[str]] = {}
        for node_id in parent:
            grouped.setdefault(find(node_id), []).append(node_id)
        result: dict[str, tuple[str, ...]] = {}
        for members in grouped.values():
            component = tuple(sorted(members))
            for node_id in component:
                result[node_id] = component
        return result

    def equivalence_class(self, node_id: str) -> tuple[str, ...]:
        self.node(node_id)
        return self._equivalence_components()[node_id]

    def proofs_for(self, node_id: str) -> tuple[str, ...]:
        self.node(node_id)
        return tuple(
            edge.source_node_id
            for edge in self.edges
            if edge.kind is KnowledgeEdgeKind.PROVES and edge.target_node_id == node_id
        )

    def derivations_for(self, node_id: str) -> tuple[str, ...]:
        self.node(node_id)
        return tuple(
            edge.source_node_id
            for edge in self.edges
            if edge.kind is KnowledgeEdgeKind.DERIVES and edge.target_node_id == node_id
        )

    def lookup_content(self, content_sha256: str) -> CorpusLookup:
        _hash(content_sha256, "content query")
        matches = tuple(
            node.node_id
            for node in self.nodes
            if node.content_sha256 == content_sha256
            or (
                node.artifact_ref is not None and node.artifact_ref.content_sha256 == content_sha256
            )
        )
        return CorpusLookup(
            queried_content_sha256=content_sha256,
            status=CorpusPresence.PRESENT if matches else CorpusPresence.ABSENT,
            node_ids=matches,
        )

    def holdout_cut(self, target_node_id: str) -> HoldoutCut:
        target_class = self.equivalence_class(target_node_id)
        equivalence_components = self._equivalence_components()
        exposing_relations: dict[str, set[str]] = {node.node_id: set() for node in self.nodes}
        for edge in self.edges:
            if edge.kind in {
                KnowledgeEdgeKind.DEPENDENCY,
                KnowledgeEdgeKind.PROVES,
                KnowledgeEdgeKind.DERIVES,
            }:
                exposing_relations[edge.target_node_id].add(edge.source_node_id)
        pending = list(target_class)
        forbidden: set[str] = set()
        while pending:
            current = pending.pop()
            for equivalent in equivalence_components[current]:
                if equivalent in forbidden:
                    continue
                forbidden.add(equivalent)
                pending.extend(exposing_relations[equivalent] - forbidden)
        forbidden_ids = tuple(sorted(forbidden))
        visible_ids = tuple(node.node_id for node in self.nodes if node.node_id not in forbidden)
        body = {
            "schema_version": HOLDOUT_SCHEMA,
            "graph_sha256": self.content_sha256,
            "target_node_id": target_node_id,
            "target_equivalence_class": list(target_class),
            "forbidden_node_ids": list(forbidden_ids),
            "visible_node_ids": list(visible_ids),
        }
        return HoldoutCut(
            graph_sha256=self.content_sha256,
            target_node_id=target_node_id,
            target_equivalence_class=target_class,
            forbidden_node_ids=forbidden_ids,
            visible_node_ids=visible_ids,
            content_sha256=canonical_sha256(body),
        )
