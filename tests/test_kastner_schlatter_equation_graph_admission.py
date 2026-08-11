from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.kastner_schlatter_equation_graph_admission as bridge
from sigma_theory_compiler.kastner_schlatter_equation_graph_admission import (
    FIRST_BLOCKER,
    SOURCE_CONTENT_SHA256,
    SOURCE_PDF_SHA256,
    _compile_formula_nodes,
    _validate_result,
    build_admission,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_equation_graph_admission.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-equation-graph-admission.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_admission(_load(CONFIG), ROOT)


def test_exact_rebuild_and_graph_counts(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision"] == "blocked"
    assert artifact["first_blocker"] == FIRST_BLOCKER
    assert artifact["graph_counts"] == {
        "nodes": 54,
        "edges": 137,
        "formula_nodes": 25,
        "assumption_nodes": 12,
        "domain_nodes": 6,
        "source_nodes": 2,
        "action_contract_nodes": 1,
        "absent_capability_nodes": 8,
        "dependency_edges": 18,
        "assumption_edges": 35,
        "semantic_algebraic_equivalence_edges": 1,
        "exact_duplicate_edges": 0,
        "theory_equivalence_edges": 0,
        "absent_action_edges": 33,
    }


def test_formula_nodes_use_existing_equation_universe_canonical_contract(
    rebuilt: dict[str, object],
) -> None:
    formulas = [
        node for node in rebuilt["knowledge_graph"]["nodes"] if node["node_type"] == "formula"
    ]
    assert len(formulas) == 25
    assert all(node["equation_universe_schema"] == "sigma-equation-universe-1.0" for node in formulas)
    assert all(node["dimension_status"] in {"pass", "unknown"} for node in formulas)
    assert all(len(node["semantic_hash"]) == 64 for node in formulas)
    assert all(len(node["record_sha256"]) == 64 for node in formulas)
    action = next(
        node
        for node in rebuilt["knowledge_graph"]["nodes"]
        if node["node_id"] == "ACTION-CONTRACT-ABSENT"
    )
    assert action["fundamental_action"] is None
    assert action["status"] == "absent"


def test_exact_duplicate_and_algebraic_equivalence_are_separate(
    rebuilt: dict[str, object],
) -> None:
    audit = rebuilt["duplicate_equivalence_audit"]
    assert audit["exact_duplicate_groups"] == []
    assert audit["exact_duplicate_group_count"] == 0
    assert audit["semantic_equivalence_not_exact_duplicate_groups"] == [
        ["EQ-KS-59-ENTROPY-QUADRATIC", "EQ-KS-59-REARRANGED-CONTROL"]
    ]
    formulas = {
        node["node_id"]: node
        for node in rebuilt["knowledge_graph"]["nodes"]
        if node["node_type"] == "formula"
    }
    original = formulas["EQ-KS-59-ENTROPY-QUADRATIC"]
    control = formulas["EQ-KS-59-REARRANGED-CONTROL"]
    assert original["exact_formula_sha256"] != control["exact_formula_sha256"]
    assert original["semantic_hash"] == control["semantic_hash"]
    equivalence_edges = [
        edge
        for edge in rebuilt["knowledge_graph"]["edges"]
        if edge["edge_type"] == "semantic_algebraic_equivalence"
    ]
    assert len(equivalence_edges) == 1
    assert {equivalence_edges[0]["source"], equivalence_edges[0]["target"]} == {
        original["node_id"],
        control["node_id"],
    }
    assert "not exact duplicate or theory equivalence" in equivalence_edges[0]["detail"]


def test_deliberate_exact_duplicate_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    records = bridge._formula_records()
    duplicate = copy.deepcopy(records[0])
    duplicate["equation_id"] = records[1]["equation_id"]
    duplicate["name"] = "renamed duplicate"
    duplicate["source_locator"] = "tamper control"
    duplicate["tags"] = ["tamper_control"]
    records[1] = duplicate
    monkeypatch.setattr(bridge, "_formula_records", lambda: records)
    with pytest.raises(ValueError, match="exact duplicate formula node admission blocked"):
        _compile_formula_nodes()


def test_absent_action_and_claim_edges_are_fail_closed(rebuilt: dict[str, object]) -> None:
    edges = rebuilt["knowledge_graph"]["edges"]
    assert sum(edge["edge_type"] == "not_derived_from_action" for edge in edges) == 25
    assert sum(edge["edge_type"] == "lacks" for edge in edges) == 8
    assert not any(
        edge["edge_type"] in {"theory_equivalent_to", "proves_gr_equivalence"} for edge in edges
    )
    assert rebuilt["admission_contract"] == {
        "kind": "equation_universe_compatible_typed_knowledge_graph",
        "equation_universe_schema": "sigma-equation-universe-1.0",
        "equation_only": True,
        "fundamental_action": None,
        "variational_edges_present": False,
        "theory_equivalence_edges_present": False,
        "observational_edges_present": False,
    }
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())


def test_source_lineage_is_exactly_bound(rebuilt: dict[str, object]) -> None:
    lineage = rebuilt["source_lineage"]
    assert lineage["source_intake_content_sha256"] == SOURCE_CONTENT_SHA256
    assert lineage["primary_pdf_sha256"] == SOURCE_PDF_SHA256
    assert lineage["source_intake_file_sha256"] == (
        "4c142f202cc30a39ad62039ae01355b91e9264260ec0ec4fd02f45a3a16f82e2"
    )
    assert rebuilt["graph_sha256"] == _sha(rebuilt["knowledge_graph"])


def test_lineage_and_artifact_tampering_fail_closed() -> None:
    config = _load(CONFIG)
    tampered_config = copy.deepcopy(config)
    tampered_config["source_intake_artifact"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source intake artifact content hash mismatch"):
        build_admission(tampered_config, ROOT)

    artifact = _load(ARTIFACT)
    tampered_artifact = copy.deepcopy(artifact)
    tampered_artifact["knowledge_graph"]["edges"].append(
        {
            "edge_id": "EDGE-TAMPER",
            "edge_type": "theory_equivalent_to",
            "source": "EQ-KS-39-TRACE-REVERSED",
            "target": "ACTION-CONTRACT-ABSENT",
            "detail": "forbidden",
        }
    )
    tampered_artifact["graph_counts"]["edges"] += 1
    tampered_artifact["content_sha256"] = _sha(
        {key: value for key, value in tampered_artifact.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="forbidden theory-equivalence edge"):
        _validate_result(tampered_artifact)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["knowledge_graph"]["nodes"][0]["record"].__setitem__(
                "expression", "p_n = 0"
            ),
            "deterministic source-bound graph",
        ),
        (
            lambda value: value.__setitem__("graph_sha256", "0" * 64),
            "registry hash",
        ),
        (
            lambda value: value["graph_counts"].__setitem__("assumption_nodes", 0),
            "equivalence partition",
        ),
        (
            lambda value: value["admission_contract"].__setitem__("fundamental_action", "S"),
            "admission contract",
        ),
        (
            lambda value: value["data_seals"].__setitem__(
                "observational_data_opened", True
            ),
            "data seal",
        ),
        (
            lambda value: value["source_lineage"].__setitem__(
                "bridge_implementation_sha256", "0" * 64
            ),
            "source lineage",
        ),
    ],
)
def test_rehashed_graph_contract_tampering_fails_closed(mutate, message: str) -> None:
    tampered = copy.deepcopy(_load(ARTIFACT))
    mutate(tampered)
    tampered["content_sha256"] = _sha(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match=message):
        _validate_result(tampered)
