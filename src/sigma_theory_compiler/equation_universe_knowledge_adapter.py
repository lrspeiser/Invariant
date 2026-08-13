"""Import sealed Equation Universe JSON into :mod:`candidate_knowledge_graph`.

The adapter reads JSON seed, policy, and audit receipts only.  It never opens the Equation
Universe SQLite database.  Query results establish presence or absence in this exact corpus and
never establish novelty, truth, proof, scientific validity, or promotion eligibility.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sigma_theory_compiler.candidate_knowledge_graph import (
    CandidateKnowledgeEdge,
    CandidateKnowledgeGraph,
    CandidateKnowledgeGraphError,
    CandidateKnowledgeNode,
    CorpusPresence,
    KnowledgeEdgeKind,
    KnowledgeGraphLimits,
    KnowledgeNodeKind,
)
from sigma_theory_compiler.equation_universe import canonicalize_record
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    ArtifactRef,
    CandidateArtifact,
    DomainPackRef,
    ProvenanceRecord,
    SchemaViolation,
    SourceBinding,
    canonical_json_bytes,
    canonical_sha256,
)

IMPORT_SCHEMA = "sigma-equation-universe-knowledge-import-1.0"
LOOKUP_SCHEMA = "sigma-equation-universe-corpus-lookup-1.0"
EQUATION_SCHEMA = "sigma-equation-universe-1.0"
POLICY_SCHEMA = "sigma-equation-source-policy-1.0"
SCOPE = (
    "Imported Equation Universe corpus membership and bound canonical equivalence only; an "
    "unmatched record is absent from this exact corpus and never establishes novelty, truth, "
    "proof, scientific validity, or promotion eligibility."
)
_IMPORT_BINDING_ROLES = ("audit_report", "equation_seed", "source_policy")
_AUDIT_KEYS = {
    "counts",
    "database",
    "derivation_proofs",
    "dimension_status",
    "formula_space_coverage",
    "integrity_check",
    "novelty_policy",
    "passed",
    "schema_version",
    "source_ingestion_modes",
    "unproven_derivations",
}
_AUDIT_COUNT_KEYS = {
    "derivations",
    "equations",
    "equivalence_edges",
    "formula_spaces",
    "import_runs",
    "sources",
    "variables",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EquationUniverseKnowledgeAdapterError(ValueError):
    """An Equation Universe import or lookup violates the fail-closed boundary."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise EquationUniverseKnowledgeAdapterError(f"{label} keys changed")


def _exact_json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EquationUniverseKnowledgeAdapterError(f"{label} must be an object")
    try:
        result = json.loads(canonical_json_bytes(value))
    except (SchemaViolation, TypeError, ValueError) as error:
        raise EquationUniverseKnowledgeAdapterError(
            f"{label} is not canonical exact JSON"
        ) from error
    if not isinstance(result, dict):
        raise EquationUniverseKnowledgeAdapterError(f"{label} must be an object")
    return result


def _load_exact_json(path: Path) -> dict[str, Any]:
    def reject_float(value: str) -> float:
        raise EquationUniverseKnowledgeAdapterError(f"floating JSON number is forbidden: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_float=reject_float)
    except EquationUniverseKnowledgeAdapterError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EquationUniverseKnowledgeAdapterError(f"cannot read exact JSON: {path}") from error
    return _exact_json_object(value, f"JSON document {path.name}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_binding(role: str, path: Path, project_root: Path) -> SourceBinding:
    try:
        relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise EquationUniverseKnowledgeAdapterError(
            f"source path escapes project root: {path}"
        ) from error
    return SourceBinding(role, relative, _file_sha256(path))


def _source_id(record: Mapping[str, Any]) -> str:
    source_id = record.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise EquationUniverseKnowledgeAdapterError("source_id must be a nonempty string")
    return source_id


def _equation_id(record: Mapping[str, Any]) -> str:
    equation_id = record.get("equation_id")
    if not isinstance(equation_id, str) or not equation_id:
        raise EquationUniverseKnowledgeAdapterError("equation_id must be a nonempty string")
    return equation_id


@dataclass(frozen=True, slots=True)
class EquationKnowledgeIndex:
    equation_id: str
    source_id: str
    semantic_hash: str
    structural_hash: str | None
    artifact_ref: ArtifactRef
    node_id: str

    def __post_init__(self) -> None:
        if not self.equation_id or not self.source_id:
            raise EquationUniverseKnowledgeAdapterError("equation index IDs must be nonempty")
        for label, value in (("semantic_hash", self.semantic_hash),):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise EquationUniverseKnowledgeAdapterError(f"{label} must be SHA-256")
        if self.structural_hash is not None and (
            not isinstance(self.structural_hash, str)
            or _SHA256.fullmatch(self.structural_hash) is None
        ):
            raise EquationUniverseKnowledgeAdapterError("structural_hash must be SHA-256 or null")
        if not isinstance(self.node_id, str) or not self.node_id.startswith("ckn-"):
            raise EquationUniverseKnowledgeAdapterError("equation index node_id is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "source_id": self.source_id,
            "semantic_hash": self.semantic_hash,
            "structural_hash": self.structural_hash,
            "artifact_ref": self.artifact_ref.to_dict(),
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EquationKnowledgeIndex:
        _exact_keys(
            value,
            {
                "equation_id",
                "source_id",
                "semantic_hash",
                "structural_hash",
                "artifact_ref",
                "node_id",
            },
            "equation knowledge index",
        )
        try:
            artifact_ref = ArtifactRef.from_dict(value["artifact_ref"])
        except (SchemaViolation, TypeError, ValueError) as error:
            raise EquationUniverseKnowledgeAdapterError(
                "equation index ArtifactRef failed"
            ) from error
        return cls(
            equation_id=str(value["equation_id"]),
            source_id=str(value["source_id"]),
            semantic_hash=str(value["semantic_hash"]),
            structural_hash=(
                None if value["structural_hash"] is None else str(value["structural_hash"])
            ),
            artifact_ref=artifact_ref,
            node_id=str(value["node_id"]),
        )


@dataclass(frozen=True, slots=True)
class EquationUniverseCorpusLookup:
    status: CorpusPresence
    semantic_hash: str
    equation_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    import_content_sha256: str
    scope: str = SCOPE
    schema_version: str = LOOKUP_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != LOOKUP_SCHEMA or self.scope != SCOPE:
            raise EquationUniverseKnowledgeAdapterError("corpus lookup schema or scope changed")
        if not isinstance(self.semantic_hash, str) or _SHA256.fullmatch(self.semantic_hash) is None:
            raise EquationUniverseKnowledgeAdapterError("lookup semantic_hash must be SHA-256")
        if self.equation_ids != tuple(sorted(set(self.equation_ids))):
            raise EquationUniverseKnowledgeAdapterError(
                "lookup equation IDs must be unique and sorted"
            )
        if self.node_ids != tuple(sorted(set(self.node_ids))):
            raise EquationUniverseKnowledgeAdapterError("lookup node IDs must be unique and sorted")
        if len(self.equation_ids) != len(self.node_ids):
            raise EquationUniverseKnowledgeAdapterError("lookup equation/node counts changed")
        expected = CorpusPresence.PRESENT if self.equation_ids else CorpusPresence.ABSENT
        if self.status is not expected:
            raise EquationUniverseKnowledgeAdapterError("lookup corpus status changed")
        if (
            not isinstance(self.import_content_sha256, str)
            or _SHA256.fullmatch(self.import_content_sha256) is None
        ):
            raise EquationUniverseKnowledgeAdapterError("lookup import hash must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "status": self.status.value,
            "semantic_hash": self.semantic_hash,
            "equation_ids": list(self.equation_ids),
            "node_ids": list(self.node_ids),
            "import_content_sha256": self.import_content_sha256,
        }


@dataclass(frozen=True, slots=True)
class EquationUniverseKnowledgeImport:
    graph: CandidateKnowledgeGraph
    source_bindings: tuple[SourceBinding, ...]
    equation_index: tuple[EquationKnowledgeIndex, ...]
    counts: Mapping[str, int]
    content_sha256: str
    scope: str = SCOPE
    schema_version: str = IMPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != IMPORT_SCHEMA or self.scope != SCOPE:
            raise EquationUniverseKnowledgeAdapterError("knowledge import schema or scope changed")
        roles = tuple(item.role for item in self.source_bindings)
        if roles != _IMPORT_BINDING_ROLES:
            raise EquationUniverseKnowledgeAdapterError("knowledge import source bindings changed")
        equation_ids = tuple(item.equation_id for item in self.equation_index)
        if equation_ids != tuple(sorted(set(equation_ids))):
            raise EquationUniverseKnowledgeAdapterError("equation index must be unique and sorted")
        counts = _exact_json_object(self.counts, "knowledge import counts")
        expected_counts = {
            "sources": len(self.graph.nodes_by_kind(KnowledgeNodeKind.DEFINITION)),
            "equations": len(self.equation_index),
            "derivations": sum(
                node.payload.get("record_kind") == "declared_derivation"
                for node in self.graph.nodes_by_kind(KnowledgeNodeKind.DERIVATION)
            ),
            "equivalence_edges": sum(
                edge.kind is KnowledgeEdgeKind.EQUIVALENCE for edge in self.graph.edges
            ),
            "artifacts": len(self.graph.artifacts),
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
        }
        if counts != expected_counts:
            raise EquationUniverseKnowledgeAdapterError("knowledge import counts changed")
        object.__setattr__(self, "counts", counts)
        node_map = {node.node_id: node for node in self.graph.nodes}
        artifact_map = {artifact.artifact_id: artifact for artifact in self.graph.artifacts}
        for item in self.equation_index:
            node = node_map.get(item.node_id)
            artifact = artifact_map.get(item.artifact_ref.artifact_id)
            if (
                node is None
                or node.kind is not KnowledgeNodeKind.ARTIFACT
                or node.payload.get("equation_id") != item.equation_id
                or node.payload.get("semantic_hash") != item.semantic_hash
                or node.artifact_ref != item.artifact_ref
                or artifact is None
                or artifact.content_sha256 != item.artifact_ref.content_sha256
            ):
                raise EquationUniverseKnowledgeAdapterError("equation index graph binding changed")
        if {item.artifact_ref.artifact_id for item in self.equation_index} != set(artifact_map):
            raise EquationUniverseKnowledgeAdapterError("equation index artifact closure changed")
        self._validate_projection_semantics(node_map, artifact_map)
        if (
            not isinstance(self.content_sha256, str)
            or _SHA256.fullmatch(self.content_sha256) is None
        ):
            raise EquationUniverseKnowledgeAdapterError("knowledge import hash must be SHA-256")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise EquationUniverseKnowledgeAdapterError("knowledge import canonical hash changed")

    def _validate_projection_semantics(
        self,
        node_map: Mapping[str, CandidateKnowledgeNode],
        artifact_map: Mapping[str, CandidateArtifact],
    ) -> None:
        source_nodes = {
            node.payload["source"]["source_id"]: node
            for node in self.graph.nodes_by_kind(KnowledgeNodeKind.DEFINITION)
            if set(node.payload) == {"record_kind", "source"}
            and node.payload["record_kind"] == "equation_universe_source"
            and isinstance(node.payload["source"], dict)
            and isinstance(node.payload["source"].get("source_id"), str)
        }
        if len(source_nodes) != self.counts["sources"]:
            raise EquationUniverseKnowledgeAdapterError("source-node projection changed")
        edge_set = {
            (edge.kind, edge.source_node_id, edge.target_node_id) for edge in self.graph.edges
        }
        equation_nodes: dict[str, CandidateKnowledgeNode] = {}
        for item in self.equation_index:
            node = node_map[item.node_id]
            artifact = artifact_map[item.artifact_ref.artifact_id]
            if set(node.payload) != {
                "record_kind",
                "equation_id",
                "source_id",
                "semantic_hash",
                "structural_hash",
                "dimension_status",
            }:
                raise EquationUniverseKnowledgeAdapterError("equation-node schema changed")
            if node.payload["record_kind"] != "equation_universe_equation":
                raise EquationUniverseKnowledgeAdapterError("equation-node record kind changed")
            if (
                item.source_id != node.payload["source_id"]
                or item.structural_hash != node.payload["structural_hash"]
                or artifact.provenance.sources != self.source_bindings
                or artifact.kind is not ArtifactKind.IDENTITY
                or artifact.claims != ("corpus_member",)
                or artifact.representation.get("equation_id") != item.equation_id
                or artifact.representation.get("source_id") != item.source_id
                or artifact.representation.get("canonical", {}).get("semantic_hash")
                != item.semantic_hash
            ):
                raise EquationUniverseKnowledgeAdapterError("equation artifact projection changed")
            source_node = source_nodes.get(item.source_id)
            if (
                source_node is None
                or (
                    KnowledgeEdgeKind.DEPENDENCY,
                    node.node_id,
                    source_node.node_id,
                )
                not in edge_set
            ):
                raise EquationUniverseKnowledgeAdapterError("equation source dependency changed")
            equation_nodes[item.equation_id] = node
        derivation_nodes = self.graph.nodes_by_kind(KnowledgeNodeKind.DERIVATION)
        declared = [
            node
            for node in derivation_nodes
            if node.payload.get("record_kind") == "declared_derivation"
        ]
        witnesses = [
            node
            for node in derivation_nodes
            if node.payload.get("record_kind") == "canonical_equivalence_witness"
        ]
        if (
            len(declared) != self.counts["derivations"]
            or len(witnesses) != self.counts["equivalence_edges"]
        ):
            raise EquationUniverseKnowledgeAdapterError("derivation projection counts changed")
        for node in declared:
            if (
                set(node.payload)
                != {
                    "record_kind",
                    "derivation",
                    "verification_status",
                    "audit_report_sha256",
                }
                or node.payload["verification_status"] != "verified_by_bound_aggregate_audit"
            ):
                raise EquationUniverseKnowledgeAdapterError("declared derivation schema changed")
            derivation = node.payload["derivation"]
            if not isinstance(derivation, dict):
                raise EquationUniverseKnowledgeAdapterError("declared derivation payload changed")
            target = equation_nodes.get(derivation.get("target_equation_id"))
            inputs = derivation.get("inputs")
            source = source_nodes.get(derivation.get("source_id"))
            if (
                target is None
                or source is None
                or not isinstance(inputs, list)
                or any(item not in equation_nodes for item in inputs)
            ):
                raise EquationUniverseKnowledgeAdapterError(
                    "declared derivation references changed"
                )
            expected_dependencies = {equation_nodes[item].node_id for item in inputs} | {
                source.node_id
            }
            actual_dependencies = {
                edge.target_node_id
                for edge in self.graph.edges
                if edge.kind is KnowledgeEdgeKind.DEPENDENCY and edge.source_node_id == node.node_id
            }
            actual_derives = {
                edge.target_node_id
                for edge in self.graph.edges
                if edge.kind is KnowledgeEdgeKind.DERIVES and edge.source_node_id == node.node_id
            }
            if actual_dependencies != expected_dependencies or actual_derives != {target.node_id}:
                raise EquationUniverseKnowledgeAdapterError("declared derivation edges changed")
        for witness in witnesses:
            if (
                set(witness.payload)
                != {
                    "record_kind",
                    "equivalence_type",
                    "left_equation_id",
                    "right_equation_id",
                    "semantic_hash",
                    "canonicalizer",
                    "audit_report_sha256",
                }
                or witness.payload["canonicalizer"] != "equation_universe.canonicalize_record"
            ):
                raise EquationUniverseKnowledgeAdapterError("equivalence witness schema changed")
            left = equation_nodes.get(witness.payload["left_equation_id"])
            right = equation_nodes.get(witness.payload["right_equation_id"])
            if left is None or right is None:
                raise EquationUniverseKnowledgeAdapterError(
                    "equivalence witness references changed"
                )
            expected = {left.node_id, right.node_id}
            dependencies = {
                edge.target_node_id
                for edge in self.graph.edges
                if edge.kind is KnowledgeEdgeKind.DEPENDENCY
                and edge.source_node_id == witness.node_id
            }
            derives = {
                edge.target_node_id
                for edge in self.graph.edges
                if edge.kind is KnowledgeEdgeKind.DERIVES and edge.source_node_id == witness.node_id
            }
            equivalence_present = any(
                edge.kind is KnowledgeEdgeKind.EQUIVALENCE
                and {edge.source_node_id, edge.target_node_id} == expected
                for edge in self.graph.edges
            )
            if dependencies != expected or derives != expected or not equivalence_present:
                raise EquationUniverseKnowledgeAdapterError("equivalence witness edges changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "graph": self.graph.to_dict(),
            "source_bindings": [item.to_dict() for item in self.source_bindings],
            "equation_index": [item.to_dict() for item in self.equation_index],
            "counts": self.counts,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EquationUniverseKnowledgeImport:
        _exact_keys(
            value,
            {
                "schema_version",
                "scope",
                "graph",
                "source_bindings",
                "equation_index",
                "counts",
                "content_sha256",
            },
            "equation universe knowledge import",
        )
        if not isinstance(value["source_bindings"], list) or not isinstance(
            value["equation_index"], list
        ):
            raise EquationUniverseKnowledgeAdapterError(
                "knowledge import bindings and index must be arrays"
            )
        try:
            graph = CandidateKnowledgeGraph.from_dict(value["graph"])
            bindings = tuple(SourceBinding.from_dict(item) for item in value["source_bindings"])
        except (CandidateKnowledgeGraphError, SchemaViolation, TypeError, ValueError) as error:
            raise EquationUniverseKnowledgeAdapterError(
                "knowledge import binding failed"
            ) from error
        return cls(
            graph=graph,
            source_bindings=bindings,
            equation_index=tuple(
                EquationKnowledgeIndex.from_dict(item) for item in value["equation_index"]
            ),
            counts=value["counts"],
            content_sha256=str(value["content_sha256"]),
            scope=str(value["scope"]),
            schema_version=str(value["schema_version"]),
        )

    def lookup_record(self, record: Mapping[str, Any]) -> EquationUniverseCorpusLookup:
        try:
            canonical = canonicalize_record(_exact_json_object(record, "equation query"))
        except EquationUniverseKnowledgeAdapterError:
            raise
        except (TypeError, ValueError) as error:
            raise EquationUniverseKnowledgeAdapterError(
                "equation query canonicalization failed closed"
            ) from error
        semantic_hash = canonical["semantic_hash"]
        matches = tuple(item for item in self.equation_index if item.semantic_hash == semantic_hash)
        return EquationUniverseCorpusLookup(
            status=CorpusPresence.PRESENT if matches else CorpusPresence.ABSENT,
            semantic_hash=semantic_hash,
            equation_ids=tuple(item.equation_id for item in matches),
            node_ids=tuple(sorted(item.node_id for item in matches)),
            import_content_sha256=self.content_sha256,
        )


def _validate_documents(
    seed: Mapping[str, Any], policy: Mapping[str, Any], audit: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    seed = _exact_json_object(seed, "equation seed")
    policy = _exact_json_object(policy, "source policy")
    audit = _exact_json_object(audit, "equation audit")
    _exact_keys(seed, {"schema_version", "sources", "equations", "derivations"}, "equation seed")
    if seed["schema_version"] != EQUATION_SCHEMA:
        raise EquationUniverseKnowledgeAdapterError("equation seed schema changed")
    for key in ("sources", "equations", "derivations"):
        if not isinstance(seed[key], list):
            raise EquationUniverseKnowledgeAdapterError(f"equation seed {key} must be an array")
    _exact_keys(
        policy,
        {"schema_version", "rules", "providers", "prohibited_uses"},
        "source policy",
    )
    if policy["schema_version"] != POLICY_SCHEMA:
        raise EquationUniverseKnowledgeAdapterError("source policy schema changed")
    if not isinstance(policy["prohibited_uses"], list) or not any(
        "unmatched equation novel" in item for item in policy["prohibited_uses"]
    ):
        raise EquationUniverseKnowledgeAdapterError("source policy novelty prohibition is absent")
    _exact_keys(audit, _AUDIT_KEYS, "equation audit")
    _exact_keys(audit["counts"], _AUDIT_COUNT_KEYS, "equation audit counts")
    _exact_keys(audit["derivation_proofs"], {"verified"}, "derivation proof counts")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in audit["counts"].values()
    ):
        raise EquationUniverseKnowledgeAdapterError("equation audit counts must be nonnegative")
    if (
        audit["schema_version"] != EQUATION_SCHEMA
        or audit["passed"] is not True
        or audit["integrity_check"] != "ok"
        or not isinstance(audit["novelty_policy"], str)
        or "absent from this corpus" not in audit["novelty_policy"]
        or "may not be labeled novel" not in audit["novelty_policy"]
    ):
        raise EquationUniverseKnowledgeAdapterError("equation audit fail-closed status changed")
    return seed, policy, audit


def import_equation_universe_payload(
    seed: Mapping[str, Any],
    policy: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    source_bindings: Sequence[SourceBinding],
    limits: KnowledgeGraphLimits | None = None,
) -> EquationUniverseKnowledgeImport:
    """Build a sealed knowledge graph from already-loaded Equation Universe documents."""

    seed, policy, audit = _validate_documents(seed, policy, audit)
    if any(not isinstance(item, SourceBinding) for item in source_bindings):
        raise EquationUniverseKnowledgeAdapterError("source bindings must be SourceBinding values")
    bindings = tuple(sorted(source_bindings, key=lambda item: item.role))
    if tuple(item.role for item in bindings) != _IMPORT_BINDING_ROLES:
        raise EquationUniverseKnowledgeAdapterError("source bindings must cover seed/policy/audit")
    binding_map = {item.role: item for item in bindings}
    sources = seed["sources"]
    source_ids = [_source_id(item) for item in sources]
    if len(set(source_ids)) != len(source_ids):
        raise EquationUniverseKnowledgeAdapterError("source IDs contain duplicates")
    source_map = {_source_id(item): item for item in sources}
    source_nodes: dict[str, CandidateKnowledgeNode] = {}
    for source in sources:
        for required in ("title", "source_kind", "ingestion_mode", "policy_reason"):
            if not isinstance(source.get(required), str) or not source[required]:
                raise EquationUniverseKnowledgeAdapterError(
                    f"source {_source_id(source)} lacks {required}"
                )
        source_nodes[_source_id(source)] = CandidateKnowledgeNode.create(
            KnowledgeNodeKind.DEFINITION,
            {"record_kind": "equation_universe_source", "source": source},
            sources=(binding_map["equation_seed"], binding_map["source_policy"]),
        )

    equations = seed["equations"]
    equation_ids = [_equation_id(item) for item in equations]
    if len(set(equation_ids)) != len(equation_ids):
        raise EquationUniverseKnowledgeAdapterError("equation IDs contain duplicates")
    audit_counts = audit["counts"]
    if (
        audit_counts["sources"] != len(sources)
        or audit_counts["equations"] != len(equations)
        or audit_counts["derivations"] != len(seed["derivations"])
        or audit_counts["variables"] != sum(len(item.get("variables", [])) for item in equations)
    ):
        raise EquationUniverseKnowledgeAdapterError("equation audit counts do not bind the seed")
    domain_pack = DomainPackRef(
        "equation.universe",
        EQUATION_SCHEMA,
        canonical_sha256(policy),
    )
    artifacts: list[CandidateArtifact] = []
    equation_nodes: dict[str, CandidateKnowledgeNode] = {}
    indexes: list[EquationKnowledgeIndex] = []
    canonical_by_equation: dict[str, dict[str, Any]] = {}
    for record in equations:
        equation_id = _equation_id(record)
        for required in ("name", "expression", "variables", "assumptions"):
            if required not in record:
                raise EquationUniverseKnowledgeAdapterError(
                    f"equation {equation_id} lacks {required}"
                )
        if (
            not isinstance(record["name"], str)
            or not isinstance(record["expression"], str)
            or not isinstance(record["variables"], list)
            or not isinstance(record["assumptions"], list)
        ):
            raise EquationUniverseKnowledgeAdapterError(
                f"equation {equation_id} has invalid required field types"
            )
        source_id = _source_id(record)
        source = source_map.get(source_id)
        if source is None:
            raise EquationUniverseKnowledgeAdapterError(
                f"equation {equation_id} has an unknown source"
            )
        if (
            source["ingestion_mode"] == "metadata_only"
            and record.get("independently_encoded") is not True
        ):
            raise EquationUniverseKnowledgeAdapterError(
                f"equation {equation_id} violates metadata-only source policy"
            )
        try:
            canonical = canonicalize_record(record)
        except (TypeError, ValueError) as error:
            raise EquationUniverseKnowledgeAdapterError(
                f"equation canonicalization failed: {equation_id}"
            ) from error
        if canonical["dimension_status"] == "fail":
            raise EquationUniverseKnowledgeAdapterError(
                f"equation dimension audit failed: {equation_id}"
            )
        canonical_by_equation[equation_id] = canonical
        provenance = ProvenanceRecord.create(
            domain_pack,
            {
                "equation_id": equation_id,
                "source_id": source_id,
                "source_record_sha256": canonical_sha256(source),
                "equation_record_sha256": canonical_sha256(record),
                "canonical_record": canonical,
                "audit_report_sha256": binding_map["audit_report"].file_sha256,
            },
            sources=bindings,
        )
        artifact = CandidateArtifact.create(
            ArtifactKind.IDENTITY,
            f"{record['name']}: {record['expression']}",
            {
                "corpus": "equation_universe",
                "equation_id": equation_id,
                "domain": record.get("domain"),
                "representation": record.get("representation", "scalar_sympy"),
                "expression": record["expression"],
                "variables": record.get("variables", []),
                "validity": record.get("validity", []),
                "tags": record.get("tags", []),
                "source_id": source_id,
                "source_locator": record.get("source_locator"),
                "canonical": canonical,
            },
            provenance,
            assumptions=tuple(sorted(set(record.get("assumptions", [])))),
            claims=("corpus_member",),
        )
        artifacts.append(artifact)
        node = CandidateKnowledgeNode.create(
            KnowledgeNodeKind.ARTIFACT,
            {
                "record_kind": "equation_universe_equation",
                "equation_id": equation_id,
                "source_id": source_id,
                "semantic_hash": canonical["semantic_hash"],
                "structural_hash": canonical["structural_hash"],
                "dimension_status": canonical["dimension_status"],
            },
            artifact_ref=artifact.ref,
            sources=bindings,
        )
        equation_nodes[equation_id] = node
        indexes.append(
            EquationKnowledgeIndex(
                equation_id=equation_id,
                source_id=source_id,
                semantic_hash=canonical["semantic_hash"],
                structural_hash=canonical["structural_hash"],
                artifact_ref=artifact.ref,
                node_id=node.node_id,
            )
        )

    dimension_counts: dict[str, int] = {}
    for canonical in canonical_by_equation.values():
        status = canonical["dimension_status"]
        dimension_counts[status] = dimension_counts.get(status, 0) + 1
    ingestion_counts: dict[str, int] = {}
    for source in sources:
        mode = source["ingestion_mode"]
        ingestion_counts[mode] = ingestion_counts.get(mode, 0) + 1
    if (
        audit["dimension_status"] != dimension_counts
        or audit["source_ingestion_modes"] != ingestion_counts
    ):
        raise EquationUniverseKnowledgeAdapterError(
            "equation audit dimension/source counts do not bind the seed"
        )

    nodes = [*source_nodes.values(), *equation_nodes.values()]
    edges = [
        CandidateKnowledgeEdge.create(
            KnowledgeEdgeKind.DEPENDENCY,
            equation_nodes[equation_id].node_id,
            source_nodes[_source_id(record)].node_id,
        )
        for record in equations
        for equation_id in (_equation_id(record),)
    ]
    derivation_ids: set[str] = set()
    verified_count = audit["derivation_proofs"].get("verified")
    if verified_count != len(seed["derivations"]) or audit["unproven_derivations"] != []:
        raise EquationUniverseKnowledgeAdapterError("derivation audit is not fully verified")
    for derivation in seed["derivations"]:
        derivation_id = derivation.get("derivation_id")
        if (
            not isinstance(derivation_id, str)
            or not derivation_id
            or derivation_id in derivation_ids
        ):
            raise EquationUniverseKnowledgeAdapterError("derivation IDs are invalid or duplicated")
        derivation_ids.add(derivation_id)
        target_id = derivation.get("target_equation_id")
        inputs = derivation.get("inputs")
        if target_id not in equation_nodes or not isinstance(inputs, list) or not inputs:
            raise EquationUniverseKnowledgeAdapterError(
                f"derivation {derivation_id} has invalid target or inputs"
            )
        if len(set(inputs)) != len(inputs) or any(item not in equation_nodes for item in inputs):
            raise EquationUniverseKnowledgeAdapterError(
                f"derivation {derivation_id} has duplicate or unknown inputs"
            )
        derivation_source = derivation.get("source_id")
        if derivation_source not in source_nodes:
            raise EquationUniverseKnowledgeAdapterError(
                f"derivation {derivation_id} has an unknown source"
            )
        node = CandidateKnowledgeNode.create(
            KnowledgeNodeKind.DERIVATION,
            {
                "record_kind": "declared_derivation",
                "derivation": derivation,
                "verification_status": "verified_by_bound_aggregate_audit",
                "audit_report_sha256": binding_map["audit_report"].file_sha256,
            },
            sources=(binding_map["audit_report"], binding_map["equation_seed"]),
        )
        nodes.append(node)
        edges.extend(
            CandidateKnowledgeEdge.create(
                KnowledgeEdgeKind.DEPENDENCY, node.node_id, equation_nodes[item].node_id
            )
            for item in inputs
        )
        edges.append(
            CandidateKnowledgeEdge.create(
                KnowledgeEdgeKind.DEPENDENCY,
                node.node_id,
                source_nodes[derivation_source].node_id,
            )
        )
        edges.append(
            CandidateKnowledgeEdge.create(
                KnowledgeEdgeKind.DERIVES, node.node_id, equation_nodes[target_id].node_id
            )
        )

    semantic_groups: dict[str, list[str]] = {}
    for equation_id, canonical in canonical_by_equation.items():
        semantic_groups.setdefault(canonical["semantic_hash"], []).append(equation_id)
    equivalence_pairs = [
        pair
        for group in semantic_groups.values()
        if len(group) > 1
        for pair in itertools.combinations(sorted(group), 2)
    ]
    if audit_counts["equivalence_edges"] != len(equivalence_pairs):
        raise EquationUniverseKnowledgeAdapterError(
            "equivalence audit count does not bind canonical semantic pairs"
        )
    for left_id, right_id in equivalence_pairs:
        left = equation_nodes[left_id]
        right = equation_nodes[right_id]
        edges.append(
            CandidateKnowledgeEdge.create(
                KnowledgeEdgeKind.EQUIVALENCE, left.node_id, right.node_id
            )
        )
        witness = CandidateKnowledgeNode.create(
            KnowledgeNodeKind.DERIVATION,
            {
                "record_kind": "canonical_equivalence_witness",
                "equivalence_type": "semantic_algebraic",
                "left_equation_id": left_id,
                "right_equation_id": right_id,
                "semantic_hash": canonical_by_equation[left_id]["semantic_hash"],
                "canonicalizer": "equation_universe.canonicalize_record",
                "audit_report_sha256": binding_map["audit_report"].file_sha256,
            },
            sources=(binding_map["audit_report"], binding_map["equation_seed"]),
        )
        nodes.append(witness)
        for equation_node in (left, right):
            edges.append(
                CandidateKnowledgeEdge.create(
                    KnowledgeEdgeKind.DEPENDENCY, witness.node_id, equation_node.node_id
                )
            )
            edges.append(
                CandidateKnowledgeEdge.create(
                    KnowledgeEdgeKind.DERIVES, witness.node_id, equation_node.node_id
                )
            )

    graph = CandidateKnowledgeGraph.create(
        "equation.universe.gravity-seed-v1",
        artifacts=artifacts,
        nodes=nodes,
        edges=edges,
        limits=limits or KnowledgeGraphLimits(),
    )
    counts = {
        "sources": len(sources),
        "equations": len(equations),
        "derivations": len(seed["derivations"]),
        "equivalence_edges": len(equivalence_pairs),
        "artifacts": len(artifacts),
        "nodes": len(nodes),
        "edges": len(edges),
    }
    provisional = object.__new__(EquationUniverseKnowledgeImport)
    fields = {
        "graph": graph,
        "source_bindings": bindings,
        "equation_index": tuple(sorted(indexes, key=lambda item: item.equation_id)),
        "counts": counts,
        "scope": SCOPE,
        "schema_version": IMPORT_SCHEMA,
    }
    for name, value in fields.items():
        object.__setattr__(provisional, name, value)
    return EquationUniverseKnowledgeImport(
        **fields, content_sha256=canonical_sha256(provisional._body())
    )


def import_equation_universe_files(
    seed_path: str | Path,
    policy_path: str | Path,
    audit_path: str | Path,
    *,
    project_root: str | Path,
    limits: KnowledgeGraphLimits | None = None,
) -> EquationUniverseKnowledgeImport:
    """Read three immutable JSON inputs and build the sealed graph without opening SQLite."""

    root = Path(project_root).resolve()
    seed_file = Path(seed_path).resolve()
    policy_file = Path(policy_path).resolve()
    audit_file = Path(audit_path).resolve()
    bindings = (
        _source_binding("audit_report", audit_file, root),
        _source_binding("equation_seed", seed_file, root),
        _source_binding("source_policy", policy_file, root),
    )
    return import_equation_universe_payload(
        _load_exact_json(seed_file),
        _load_exact_json(policy_file),
        _load_exact_json(audit_file),
        source_bindings=bindings,
        limits=limits,
    )
