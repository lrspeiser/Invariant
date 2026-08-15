from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.candidate_knowledge_graph import (
    SCOPE,
    CandidateKnowledgeEdge,
    CandidateKnowledgeGraph,
    CandidateKnowledgeGraphError,
    CandidateKnowledgeNode,
    CorpusPresence,
    HoldoutCut,
    KnowledgeEdgeKind,
    KnowledgeGraphLimits,
    KnowledgeNodeKind,
)
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    DomainPackRef,
    ProvenanceRecord,
    SourceBinding,
    canonical_sha256,
)

PACK = DomainPackRef("graph.test", "1.0", "3" * 64)
SOURCE = SourceBinding("definition_source", "sources/definitions.json", "4" * 64)


def _artifact() -> CandidateArtifact:
    return CandidateArtifact.create(
        ArtifactKind.CONJECTURE,
        "bounded graph candidate",
        {"formula": "a+b"},
        ProvenanceRecord.create(PACK, {"campaign": "graph-test"}),
        claims=("candidate_formula",),
    )


def _node(
    label: str,
    kind: KnowledgeNodeKind,
    *,
    artifact: CandidateArtifact | None = None,
) -> CandidateKnowledgeNode:
    return CandidateKnowledgeNode.create(
        kind,
        {"label": label},
        artifact_ref=None if artifact is None else artifact.ref,
        sources=(SOURCE,) if kind is KnowledgeNodeKind.DEFINITION else (),
    )


def _components() -> tuple[
    CandidateArtifact,
    dict[str, CandidateKnowledgeNode],
    tuple[CandidateKnowledgeEdge, ...],
]:
    artifact = _artifact()
    nodes = {
        "artifact": _node("artifact", KnowledgeNodeKind.ARTIFACT, artifact=artifact),
        "definition": _node("definition", KnowledgeNodeKind.DEFINITION),
        "axiom": _node("axiom", KnowledgeNodeKind.AXIOM),
        "lemma": _node("lemma", KnowledgeNodeKind.LEMMA),
        "theorem": _node("theorem", KnowledgeNodeKind.THEOREM),
        "theorem_equivalent": _node("theorem equivalent", KnowledgeNodeKind.THEOREM),
        "conjecture": _node("conjecture", KnowledgeNodeKind.CONJECTURE),
        "identity": _node("identity", KnowledgeNodeKind.IDENTITY),
        "construction": _node("construction", KnowledgeNodeKind.CONSTRUCTION),
        "algorithm": _node("algorithm", KnowledgeNodeKind.ALGORITHM),
        "counterexample": _node("counterexample", KnowledgeNodeKind.COUNTEREXAMPLE),
        "proof": _node("proof", KnowledgeNodeKind.PROOF),
        "derivation": _node("derivation", KnowledgeNodeKind.DERIVATION),
    }

    def edge(kind: KnowledgeEdgeKind, source: str, target: str) -> CandidateKnowledgeEdge:
        return CandidateKnowledgeEdge.create(kind, nodes[source].node_id, nodes[target].node_id)

    edges = (
        edge(KnowledgeEdgeKind.DEPENDENCY, "definition", "axiom"),
        edge(KnowledgeEdgeKind.DEPENDENCY, "lemma", "axiom"),
        edge(KnowledgeEdgeKind.DEPENDENCY, "theorem", "lemma"),
        edge(KnowledgeEdgeKind.DEPENDENCY, "construction", "theorem_equivalent"),
        edge(KnowledgeEdgeKind.DEPENDENCY, "algorithm", "construction"),
        edge(KnowledgeEdgeKind.DEPENDENCY, "counterexample", "conjecture"),
        edge(KnowledgeEdgeKind.DEPENDENCY, "proof", "lemma"),
        edge(KnowledgeEdgeKind.DEPENDENCY, "derivation", "axiom"),
        edge(KnowledgeEdgeKind.EQUIVALENCE, "theorem", "theorem_equivalent"),
        edge(KnowledgeEdgeKind.PROVES, "proof", "theorem"),
        edge(KnowledgeEdgeKind.DERIVES, "derivation", "identity"),
    )
    return artifact, nodes, edges


def _graph(limits: KnowledgeGraphLimits | None = None) -> CandidateKnowledgeGraph:
    artifact, nodes, edges = _components()
    return CandidateKnowledgeGraph.create(
        "unit.corpus",
        artifacts=(artifact,),
        nodes=tuple(nodes.values()),
        edges=edges,
        limits=limits or KnowledgeGraphLimits(4, 32, 64, 16, 8),
    )


def test_closed_node_and_edge_kinds_sigma_bindings_and_exact_ids() -> None:
    graph = _graph()

    assert {node.kind for node in graph.nodes} == set(KnowledgeNodeKind)
    assert {edge.kind for edge in graph.edges} == set(KnowledgeEdgeKind)
    assert all(node.node_id == f"ckn-{node.content_sha256[:24]}" for node in graph.nodes)
    assert all(edge.edge_id == f"cke-{edge.content_sha256[:24]}" for edge in graph.edges)
    artifact_nodes = graph.nodes_by_kind(KnowledgeNodeKind.ARTIFACT)
    assert artifact_nodes[0].artifact_ref == graph.artifacts[0].ref
    definition = graph.nodes_by_kind(KnowledgeNodeKind.DEFINITION)[0]
    assert definition.sources == (SOURCE,)
    assert graph.scope == SCOPE


def test_dependency_queries_proof_derivation_and_symmetric_equivalence() -> None:
    graph = _graph()
    _, nodes, _ = _components()

    theorem_closure = graph.dependency_closure((nodes["theorem"].node_id,))
    assert theorem_closure == tuple(
        sorted(
            {
                nodes["theorem"].node_id,
                nodes["lemma"].node_id,
                nodes["axiom"].node_id,
            }
        )
    )
    equivalence = graph.equivalence_class(nodes["theorem"].node_id)
    assert equivalence == graph.equivalence_class(nodes["theorem_equivalent"].node_id)
    assert equivalence == tuple(
        sorted((nodes["theorem"].node_id, nodes["theorem_equivalent"].node_id))
    )
    assert graph.proofs_for(nodes["theorem"].node_id) == (nodes["proof"].node_id,)
    assert graph.derivations_for(nodes["identity"].node_id) == (nodes["derivation"].node_id,)


def test_holdout_cut_forbids_target_equivalence_class_and_all_dependency_downstream() -> None:
    graph = _graph()
    _, nodes, _ = _components()

    cut = graph.holdout_cut(nodes["theorem"].node_id)

    expected_forbidden = {
        nodes["theorem"].node_id,
        nodes["theorem_equivalent"].node_id,
        nodes["construction"].node_id,
        nodes["algorithm"].node_id,
        nodes["proof"].node_id,
    }
    assert set(cut.forbidden_node_ids) == expected_forbidden
    assert not expected_forbidden & set(cut.visible_node_ids)
    assert set(cut.forbidden_node_ids) | set(cut.visible_node_ids) == {
        node.node_id for node in graph.nodes
    }
    assert cut.graph_sha256 == graph.content_sha256
    assert HoldoutCut.from_dict(cut.to_dict()) == cut
    cut.validate_against(graph)


def test_content_lookup_reports_corpus_presence_or_corpus_absence_never_novelty() -> None:
    graph = _graph()
    artifact = graph.artifacts[0]

    present = graph.lookup_content(artifact.content_sha256)
    absent = graph.lookup_content("f" * 64)

    assert present.status is CorpusPresence.PRESENT
    assert present.node_ids == (graph.nodes_by_kind(KnowledgeNodeKind.ARTIFACT)[0].node_id,)
    assert absent.status is CorpusPresence.ABSENT
    assert absent.node_ids == ()
    assert absent.to_dict()["status"] == "absent_from_this_corpus"
    assert "novel" not in absent.to_dict()
    assert "not a novelty claim" in absent.to_dict()["scope"]


def test_serialization_and_queries_are_deterministic_under_input_permutation() -> None:
    artifact, nodes, edges = _components()
    limits = KnowledgeGraphLimits(4, 32, 64, 16, 8)
    forward = CandidateKnowledgeGraph.create(
        "unit.corpus",
        artifacts=(artifact,),
        nodes=tuple(nodes.values()),
        edges=edges,
        limits=limits,
    )
    reversed_input = CandidateKnowledgeGraph.create(
        "unit.corpus",
        artifacts=(artifact,),
        nodes=tuple(reversed(tuple(nodes.values()))),
        edges=tuple(reversed(edges)),
        limits=limits,
    )

    assert forward == reversed_input
    assert forward.content_sha256 == reversed_input.content_sha256
    assert CandidateKnowledgeGraph.from_dict(forward.to_dict()) == forward


def test_unknown_dangling_and_incompatible_proof_derivation_edges_fail_closed() -> None:
    graph = _graph()
    with pytest.raises(CandidateKnowledgeGraphError, match="unknown knowledge node"):
        graph.node("missing.node")

    artifact, nodes, edges = _components()
    dangling = CandidateKnowledgeEdge.create(
        KnowledgeEdgeKind.DEPENDENCY, nodes["lemma"].node_id, "missing.node"
    )
    with pytest.raises(CandidateKnowledgeGraphError, match="dangling endpoint"):
        CandidateKnowledgeGraph.create(
            "bad.corpus",
            artifacts=(artifact,),
            nodes=tuple(nodes.values()),
            edges=(*edges, dangling),
        )

    wrong_proof = CandidateKnowledgeEdge.create(
        KnowledgeEdgeKind.PROVES, nodes["lemma"].node_id, nodes["theorem"].node_id
    )
    with pytest.raises(CandidateKnowledgeGraphError, match="proof edge"):
        CandidateKnowledgeGraph.create(
            "bad.corpus",
            artifacts=(artifact,),
            nodes=tuple(nodes.values()),
            edges=(*edges, wrong_proof),
        )

    wrong_derivation = CandidateKnowledgeEdge.create(
        KnowledgeEdgeKind.DERIVES, nodes["proof"].node_id, nodes["identity"].node_id
    )
    with pytest.raises(CandidateKnowledgeGraphError, match="derivation edge"):
        CandidateKnowledgeGraph.create(
            "bad.corpus",
            artifacts=(artifact,),
            nodes=tuple(nodes.values()),
            edges=(*edges, wrong_derivation),
        )


def test_dependency_cycles_depth_equivalence_and_capacity_limits_fail_closed() -> None:
    first = _node("first", KnowledgeNodeKind.AXIOM)
    second = _node("second", KnowledgeNodeKind.LEMMA)
    third = _node("third", KnowledgeNodeKind.THEOREM)
    cycle_edges = (
        CandidateKnowledgeEdge.create(KnowledgeEdgeKind.DEPENDENCY, first.node_id, second.node_id),
        CandidateKnowledgeEdge.create(KnowledgeEdgeKind.DEPENDENCY, second.node_id, first.node_id),
    )
    with pytest.raises(CandidateKnowledgeGraphError, match="cycle"):
        CandidateKnowledgeGraph.create(
            "cycle.corpus", artifacts=(), nodes=(first, second), edges=cycle_edges
        )

    chain_edges = (
        CandidateKnowledgeEdge.create(KnowledgeEdgeKind.DEPENDENCY, second.node_id, first.node_id),
        CandidateKnowledgeEdge.create(KnowledgeEdgeKind.DEPENDENCY, third.node_id, second.node_id),
    )
    with pytest.raises(CandidateKnowledgeGraphError, match="depth limit"):
        CandidateKnowledgeGraph.create(
            "deep.corpus",
            artifacts=(),
            nodes=(first, second, third),
            edges=chain_edges,
            limits=KnowledgeGraphLimits(1, 3, 3, 2, 3),
        )

    equivalence = CandidateKnowledgeEdge.create(
        KnowledgeEdgeKind.EQUIVALENCE, first.node_id, second.node_id
    )
    with pytest.raises(CandidateKnowledgeGraphError, match="equivalence class size"):
        CandidateKnowledgeGraph.create(
            "equivalence.corpus",
            artifacts=(),
            nodes=(first, second),
            edges=(equivalence,),
            limits=KnowledgeGraphLimits(1, 2, 2, 2, 1),
        )

    with pytest.raises(CandidateKnowledgeGraphError, match="node limit"):
        CandidateKnowledgeGraph.create(
            "small.corpus",
            artifacts=(),
            nodes=(first, second),
            edges=(),
            limits=KnowledgeGraphLimits(1, 1, 1, 1, 1),
        )


def test_float_payload_dangling_artifact_and_self_edges_fail_closed() -> None:
    with pytest.raises(CandidateKnowledgeGraphError, match="canonical exact JSON"):
        CandidateKnowledgeNode.create(KnowledgeNodeKind.AXIOM, {"weight": 0.5})
    with pytest.raises(CandidateKnowledgeGraphError, match="requires an ArtifactRef"):
        CandidateKnowledgeNode.create(KnowledgeNodeKind.ARTIFACT, {"label": "missing"})

    artifact = _artifact()
    artifact_node = _node("artifact", KnowledgeNodeKind.ARTIFACT, artifact=artifact)
    with pytest.raises(CandidateKnowledgeGraphError, match="dangling or changed"):
        CandidateKnowledgeGraph.create(
            "dangling.artifact",
            artifacts=(),
            nodes=(artifact_node,),
            edges=(),
        )
    with pytest.raises(CandidateKnowledgeGraphError, match="self-loop"):
        CandidateKnowledgeEdge.create(
            KnowledgeEdgeKind.DEPENDENCY, artifact_node.node_id, artifact_node.node_id
        )


def test_nested_and_resealed_semantic_tamper_fail_closed() -> None:
    graph = _graph()
    nested = copy.deepcopy(graph.to_dict())
    nested["nodes"][0]["payload"]["label"] = "tampered"
    with pytest.raises(CandidateKnowledgeGraphError, match="node canonical identity"):
        CandidateKnowledgeGraph.from_dict(nested)

    unknown = copy.deepcopy(graph.to_dict())
    unknown["novelty"] = True
    with pytest.raises(CandidateKnowledgeGraphError, match="keys changed"):
        CandidateKnowledgeGraph.from_dict(unknown)

    resealed = copy.deepcopy(graph.to_dict())
    resealed["scope"] = "corpus absence proves novelty"
    body = {key: value for key, value in resealed.items() if key != "content_sha256"}
    resealed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(CandidateKnowledgeGraphError, match="schema or scope"):
        CandidateKnowledgeGraph.from_dict(resealed)

    cut = graph.holdout_cut(graph.nodes_by_kind(KnowledgeNodeKind.THEOREM)[0].node_id)
    cut_tamper = copy.deepcopy(cut.to_dict())
    cut_tamper["visible_node_ids"] = sorted({*cut_tamper["visible_node_ids"], cut.target_node_id})
    cut_body = {key: value for key, value in cut_tamper.items() if key != "content_sha256"}
    cut_tamper["content_sha256"] = canonical_sha256(cut_body)
    with pytest.raises(CandidateKnowledgeGraphError, match="overlap"):
        HoldoutCut.from_dict(cut_tamper)

    incomplete = copy.deepcopy(cut.to_dict())
    removed = next(
        node_id
        for node_id in incomplete["forbidden_node_ids"]
        if node_id not in incomplete["target_equivalence_class"]
    )
    incomplete["forbidden_node_ids"].remove(removed)
    incomplete["visible_node_ids"] = sorted({*incomplete["visible_node_ids"], removed})
    incomplete_body = {key: value for key, value in incomplete.items() if key != "content_sha256"}
    incomplete["content_sha256"] = canonical_sha256(incomplete_body)
    resealed_cut = HoldoutCut.from_dict(incomplete)
    with pytest.raises(CandidateKnowledgeGraphError, match="complete target downstream"):
        resealed_cut.validate_against(graph)
