from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.candidate_knowledge_graph import (
    CorpusPresence,
    KnowledgeEdgeKind,
    KnowledgeNodeKind,
)
from sigma_theory_compiler.equation_universe_knowledge_adapter import (
    SCOPE,
    EquationUniverseKnowledgeAdapterError,
    EquationUniverseKnowledgeImport,
    import_equation_universe_files,
    import_equation_universe_payload,
)
from sigma_theory_compiler.sigma_core import ArtifactKind, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "configs" / "equation_universe" / "gravity_seed_v1.json"
POLICY = ROOT / "configs" / "equation_universe" / "source_policy.json"
AUDIT = ROOT / "runs" / "equation-universe" / "audit-report.json"


@pytest.fixture(scope="module")
def imported() -> EquationUniverseKnowledgeImport:
    return import_equation_universe_files(SEED, POLICY, AUDIT, project_root=ROOT)


def _force_record(expression: str) -> dict[str, object]:
    return {
        "equation_id": "QUERY",
        "name": "query",
        "representation": "scalar_sympy",
        "expression": expression,
        "variables": [
            {
                "symbol": "F",
                "canonical_name": "F",
                "meaning": "force",
                "dimension": {"M": 1, "L": 1, "T": -2},
            },
            {
                "symbol": "m",
                "canonical_name": "m",
                "meaning": "mass",
                "dimension": {"M": 1},
            },
            {
                "symbol": "a",
                "canonical_name": "a",
                "meaning": "acceleration",
                "dimension": {"L": 1, "T": -2},
            },
        ],
    }


def test_static_import_has_exact_counts_sigma_artifacts_and_no_sqlite_dependency(
    imported: EquationUniverseKnowledgeImport,
) -> None:
    assert imported.counts == {
        "sources": 9,
        "equations": 18,
        "derivations": 3,
        "equivalence_edges": 1,
        "artifacts": 18,
        "nodes": 31,
        "edges": 36,
    }
    assert len(imported.graph.artifacts) == 18
    assert all(artifact.kind is ArtifactKind.IDENTITY for artifact in imported.graph.artifacts)
    assert all(artifact.claims == ("corpus_member",) for artifact in imported.graph.artifacts)
    assert imported.scope == SCOPE
    assert [item.role for item in imported.source_bindings] == [
        "audit_report",
        "equation_seed",
        "source_policy",
    ]
    assert all(not item.path.endswith(".sqlite") for item in imported.source_bindings)


def test_source_derivation_and_equivalence_provenance_are_preserved(
    imported: EquationUniverseKnowledgeImport,
) -> None:
    source_nodes = imported.graph.nodes_by_kind(KnowledgeNodeKind.DEFINITION)
    derivation_nodes = imported.graph.nodes_by_kind(KnowledgeNodeKind.DERIVATION)
    declared = [
        node for node in derivation_nodes if node.payload["record_kind"] == "declared_derivation"
    ]
    witnesses = [
        node
        for node in derivation_nodes
        if node.payload["record_kind"] == "canonical_equivalence_witness"
    ]

    assert len(source_nodes) == 9
    assert all(len(node.sources) == 2 for node in source_nodes)
    assert len(declared) == 3
    assert all(
        node.payload["verification_status"] == "verified_by_bound_aggregate_audit"
        for node in declared
    )
    assert len(witnesses) == 1
    assert witnesses[0].payload["equivalence_type"] == "semantic_algebraic"
    assert witnesses[0].payload["canonicalizer"] == "equation_universe.canonicalize_record"

    equivalence_edges = [
        edge for edge in imported.graph.edges if edge.kind is KnowledgeEdgeKind.EQUIVALENCE
    ]
    assert len(equivalence_edges) == 1
    equivalence = equivalence_edges[0]
    assert imported.graph.equivalence_class(equivalence.source_node_id) == tuple(
        sorted((equivalence.source_node_id, equivalence.target_node_id))
    )
    assert sum(edge.kind is KnowledgeEdgeKind.DERIVES for edge in imported.graph.edges) == 5


def test_semantic_lookup_returns_only_corpus_presence_or_absence(
    imported: EquationUniverseKnowledgeImport,
) -> None:
    present = imported.lookup_record(_force_record("F - m*a = 0"))
    absent = imported.lookup_record(
        {
            "equation_id": "QUERY-ABSENT",
            "name": "absent query",
            "representation": "scalar_sympy",
            "expression": "y = exp(x)",
            "variables": [
                {"symbol": "x", "canonical_name": "x", "dimension": {}},
                {"symbol": "y", "canonical_name": "y", "dimension": {}},
            ],
        }
    )

    assert present.status is CorpusPresence.PRESENT
    assert present.equation_ids == (
        "EQ-NEWTON-SECOND-LAW",
        "EQ-NEWTON-SECOND-LAW-REARRANGED",
    )
    assert absent.status is CorpusPresence.ABSENT
    assert absent.equation_ids == ()
    assert absent.to_dict()["status"] == "absent_from_this_corpus"
    assert "novelty" not in absent.to_dict()
    assert "never establishes novelty" in absent.scope


def test_import_serialization_is_closed_deterministic_and_replayable(
    imported: EquationUniverseKnowledgeImport,
) -> None:
    replayed = EquationUniverseKnowledgeImport.from_dict(imported.to_dict())

    assert replayed == imported
    assert replayed.content_sha256 == imported.content_sha256
    assert replayed.graph.content_sha256 == imported.graph.content_sha256
    assert [item.equation_id for item in imported.equation_index] == sorted(
        item.equation_id for item in imported.equation_index
    )


def test_file_import_is_byte_bound_and_rejects_paths_outside_project_root(
    imported: EquationUniverseKnowledgeImport,
) -> None:
    assert imported.source_bindings[0].file_sha256 == canonical_file_hash(AUDIT)
    assert imported.source_bindings[1].file_sha256 == canonical_file_hash(SEED)
    assert imported.source_bindings[2].file_sha256 == canonical_file_hash(POLICY)

    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="escapes project root"):
        import_equation_universe_files(SEED, POLICY, AUDIT, project_root=ROOT / "configs")


def canonical_file_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_policy_audit_count_unknown_source_and_float_inputs_fail_closed(
    imported: EquationUniverseKnowledgeImport,
) -> None:
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    bad_policy = copy.deepcopy(policy)
    bad_policy["prohibited_uses"] = []
    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="novelty prohibition"):
        import_equation_universe_payload(
            seed, bad_policy, audit, source_bindings=imported.source_bindings
        )

    bad_audit = copy.deepcopy(audit)
    bad_audit["counts"]["equations"] = 17
    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="counts do not bind"):
        import_equation_universe_payload(
            seed, policy, bad_audit, source_bindings=imported.source_bindings
        )

    bad_seed = copy.deepcopy(seed)
    bad_seed["equations"][0]["source_id"] = "SRC-MISSING"
    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="unknown source"):
        import_equation_universe_payload(
            bad_seed, policy, audit, source_bindings=imported.source_bindings
        )

    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="canonical exact JSON"):
        imported.lookup_record({"expression": "x = 1", "weight": 0.5})


def test_nested_and_resealed_import_tampering_fails_closed(
    imported: EquationUniverseKnowledgeImport,
) -> None:
    nested = copy.deepcopy(imported.to_dict())
    nested["equation_index"][0]["semantic_hash"] = "0" * 64
    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="index graph binding"):
        EquationUniverseKnowledgeImport.from_dict(nested)

    unknown = copy.deepcopy(imported.to_dict())
    unknown["novelty_claim_allowed"] = True
    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="keys changed"):
        EquationUniverseKnowledgeImport.from_dict(unknown)

    resealed = copy.deepcopy(imported.to_dict())
    resealed["scope"] = "absence proves novelty"
    body = {key: value for key, value in resealed.items() if key != "content_sha256"}
    resealed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="schema or scope"):
        EquationUniverseKnowledgeImport.from_dict(resealed)

    graph_tamper = copy.deepcopy(imported.to_dict())
    graph_tamper["graph"]["nodes"][0]["payload"]["record_kind"] = "forged"
    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="binding failed"):
        EquationUniverseKnowledgeImport.from_dict(graph_tamper)

    resealed_edges = copy.deepcopy(imported.to_dict())
    derivation_node_id = next(
        node["node_id"]
        for node in resealed_edges["graph"]["nodes"]
        if node["kind"] == "derivation"
        and node["payload"].get("record_kind") == "declared_derivation"
    )
    removed_edge = next(
        edge
        for edge in resealed_edges["graph"]["edges"]
        if edge["kind"] == "dependency" and edge["source_node_id"] == derivation_node_id
    )
    resealed_edges["graph"]["edges"].remove(removed_edge)
    graph_body = {
        key: value for key, value in resealed_edges["graph"].items() if key != "content_sha256"
    }
    resealed_edges["graph"]["content_sha256"] = canonical_sha256(graph_body)
    resealed_edges["counts"]["edges"] -= 1
    import_body = {key: value for key, value in resealed_edges.items() if key != "content_sha256"}
    resealed_edges["content_sha256"] = canonical_sha256(import_body)
    with pytest.raises(EquationUniverseKnowledgeAdapterError, match="derivation edges changed"):
        EquationUniverseKnowledgeImport.from_dict(resealed_edges)
