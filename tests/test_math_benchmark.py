from __future__ import annotations

import pytest

from sigma_theory_compiler.math_benchmark import (
    BenchmarkBoundaryError,
    BenchmarkTally,
    BlindHoldoutManifest,
    DiscoveryScore,
    KnowledgeNode,
    MathematicalKnowledgeGraph,
    MathObjectKind,
)
from sigma_theory_compiler.sigma_core import canonical_sha256


def node(
    node_id: str,
    kind: MathObjectKind,
    dependencies: tuple[str, ...] = (),
) -> KnowledgeNode:
    return KnowledgeNode(node_id, kind, canonical_sha256({"node_id": node_id}), dependencies)


def graph() -> MathematicalKnowledgeGraph:
    return MathematicalKnowledgeGraph(
        (
            node("axiom.naturals", MathObjectKind.AXIOM),
            node("definition.sum", MathObjectKind.DEFINITION, ("axiom.naturals",)),
            node("lemma.recurrence", MathObjectKind.LEMMA, ("definition.sum",)),
            node("theorem.natural-sum", MathObjectKind.THEOREM, ("lemma.recurrence",)),
            node(
                "theorem.natural-sum-corollary",
                MathObjectKind.THEOREM,
                ("theorem.natural-sum",),
            ),
        )
    )


def manifest(selected: MathematicalKnowledgeGraph | None = None) -> BlindHoldoutManifest:
    selected = selected or graph()
    return BlindHoldoutManifest(
        benchmark_id="natural-sum",
        target_node_id="theorem.natural-sum",
        knowledge_graph_sha256=selected.graph_sha256,
        generation_visible=("axiom.naturals", "definition.sum"),
        verification_allowed=("lemma.recurrence",),
        pre_unseal_forbidden=("theorem.natural-sum", "theorem.natural-sum-corollary"),
        post_unseal_visible=("theorem.natural-sum",),
    )


def test_graph_has_stable_hash_and_exact_dependency_closure() -> None:
    selected = graph()
    assert len(selected.graph_sha256) == 64
    assert selected.dependency_closure(("theorem.natural-sum",)) == (
        "axiom.naturals",
        "definition.sum",
        "lemma.recurrence",
        "theorem.natural-sum",
    )
    assert selected.descendants("theorem.natural-sum") == (
        "theorem.natural-sum",
        "theorem.natural-sum-corollary",
    )


def test_manifest_binds_graph_and_permits_only_allowed_proof_closure() -> None:
    selected = graph()
    holdout = manifest(selected)
    holdout.validate_against(selected)
    assert len(holdout.manifest_sha256) == 64
    assert holdout.check_proof_dependencies(selected, ("lemma.recurrence",)) == (
        "axiom.naturals",
        "definition.sum",
        "lemma.recurrence",
    )


@pytest.mark.parametrize(
    "dependencies",
    [
        ("theorem.natural-sum",),
        ("theorem.natural-sum-corollary",),
    ],
)
def test_target_or_downstream_dependency_fails_closed(dependencies: tuple[str, ...]) -> None:
    with pytest.raises(BenchmarkBoundaryError, match="escapes"):
        manifest().check_proof_dependencies(graph(), dependencies)


def test_manifest_rejects_target_leak_and_missing_downstream_forbidden_node() -> None:
    selected = graph()
    with pytest.raises(BenchmarkBoundaryError, match="exposed"):
        BlindHoldoutManifest(
            "bad-holdout",
            "theorem.natural-sum",
            selected.graph_sha256,
            ("theorem.natural-sum",),
            (),
            ("theorem.natural-sum", "theorem.natural-sum-corollary"),
            ("theorem.natural-sum",),
        )
    incomplete = BlindHoldoutManifest(
        "bad-holdout",
        "theorem.natural-sum",
        selected.graph_sha256,
        ("axiom.naturals",),
        ("lemma.recurrence",),
        ("theorem.natural-sum",),
        ("theorem.natural-sum",),
    )
    with pytest.raises(BenchmarkBoundaryError, match="downstream"):
        incomplete.validate_against(selected)


def test_graph_rejects_cycles_unknown_dependencies_and_resealed_hash_drift() -> None:
    with pytest.raises(BenchmarkBoundaryError, match="cycle"):
        MathematicalKnowledgeGraph(
            (
                node("lemma.a", MathObjectKind.LEMMA, ("lemma.b",)),
                node("lemma.b", MathObjectKind.LEMMA, ("lemma.a",)),
            )
        )
    with pytest.raises(BenchmarkBoundaryError, match="unregistered"):
        MathematicalKnowledgeGraph((node("lemma.a", MathObjectKind.LEMMA, ("lemma.missing",)),))
    selected = graph()
    rebound = BlindHoldoutManifest(
        "natural-sum",
        "theorem.natural-sum",
        "f" * 64,
        ("axiom.naturals", "definition.sum"),
        ("lemma.recurrence",),
        ("theorem.natural-sum", "theorem.natural-sum-corollary"),
        ("theorem.natural-sum",),
    )
    with pytest.raises(BenchmarkBoundaryError, match="binding"):
        rebound.validate_against(selected)


def test_hard_gates_dominate_simplicity_and_never_compensate() -> None:
    proved = DiscoveryScore(True, True, True, True, True, True, True, 50)
    unproved_but_short = DiscoveryScore(True, True, True, True, False, True, True, 1)
    assert proved.discovered_and_proved
    assert not unproved_but_short.discovered_and_proved
    assert proved.lexicographic_key > unproved_but_short.lexicographic_key


def test_tally_reports_an_exact_rate_and_rejects_impossible_counts() -> None:
    tally = BenchmarkTally(200, 130, 120, 115, 5, 2)
    assert tally.proved_rate == {"numerator": 115, "denominator": 200}
    with pytest.raises(BenchmarkBoundaryError, match="inconsistent"):
        BenchmarkTally(10, 8, 9, 7, 0, 0)
    with pytest.raises(BenchmarkBoundaryError, match="exceed"):
        BenchmarkTally(10, 8, 8, 7, 4, 0)
