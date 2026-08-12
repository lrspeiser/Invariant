from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.sigma_core import (
    CandidateArtifact,
    GateOutcome,
    PromotionLedger,
    StageOutcome,
)
from sigma_theory_compiler.synthetic_graph_cut_holdout_world import (
    CANONICAL_CUT_COUNT,
    CLAIMS,
    CONFIG_PATH,
    EDGE_COUNT,
    EXPECTED_CONFIG,
    OUTPUT_PATH,
    PACK_DESCRIPTOR,
    SOURCE_PATH,
    TEST_PATH,
    VERTEX_COUNT,
    _boundary_edges,
    _canonical_cuts,
    _cut_classes,
    _discover,
    _inside,
    _reference_world,
    _validate_config,
    build_benchmark,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_PATH
ARTIFACT = ROOT / OUTPUT_PATH


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _reseal(value: dict[str, object]) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


@pytest.fixture(scope="module")
def benchmark() -> dict[str, object]:
    return build_benchmark(CONFIG)


def test_exact_benchmark_matches_immutable_artifact_and_replays(
    benchmark: dict[str, object],
) -> None:
    checked = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert benchmark == checked == build_benchmark(CONFIG)
    body = {key: item for key, item in checked.items() if key != "content_sha256"}
    assert checked["content_sha256"] == hashlib.sha256(_canonical(body)).hexdigest()
    validate_result(checked, root=ROOT)


def test_fresh_graph_axioms_and_reference_graph_are_exact(
    benchmark: dict[str, object],
) -> None:
    world = benchmark["world"]
    graph = benchmark["reference_graph"]
    assert world["generation_epoch"] == "2026-08-12"
    assert world["vertex_count"] == VERTEX_COUNT
    assert world["edge_count"] == EDGE_COUNT
    assert len(world["vertices"]) == len(set(world["vertices"])) == VERTEX_COUNT
    edges = [tuple(edge) for edge in world["edges"]]
    assert len(edges) == len(set(edges)) == EDGE_COUNT
    assert all(left < right for left, right in edges)
    reached = {world["vertices"][0]}
    while True:
        expanded = reached | {
            right if left in reached else left
            for left, right in edges
            if (left in reached) != (right in reached)
        }
        if expanded == reached:
            break
        reached = expanded
    assert reached == set(world["vertices"])
    assert graph["axiom_count"] == 1
    assert graph["visible_ancestor_count"] == 2
    assert graph["hidden_cut_invariant_class_count"] == 1
    assert graph["visible_cut_invariant_class_count"] + 1 == graph["cut_invariant_class_count"]


def test_complete_cut_enumeration_ancestors_and_hidden_class_rediscovery(
    benchmark: dict[str, object],
) -> None:
    world = benchmark["world"]
    cuts = _canonical_cuts(world["vertices"])
    assert len(cuts) == len(set(cuts)) == CANONICAL_CUT_COUNT == 255
    full = set(world["vertices"])
    for subset in cuts:
        complement = tuple(sorted(full - set(subset)))
        assert subset != complement
        assert _boundary_edges(subset, world["edges"]) == _boundary_edges(
            complement, world["edges"]
        )
    classes = _cut_classes(world["vertices"], world["edges"])
    assert sum(row["member_count"] for row in classes) == CANONICAL_CUT_COUNT
    target = benchmark["post_unseal"]["target"]
    candidate = benchmark["rediscovery"]["candidate_artifact"]["representation"]
    assert candidate == {
        key: target[key]
        for key in (
            "class_id",
            "class_root_sha256",
            "cut_size",
            "member_count",
            "members",
            "member_root_sha256",
            "canonical_cuts_checked",
        )
    }
    assert target["member_count"] >= 1


def test_discovery_receives_only_public_subgraph_not_target() -> None:
    reference = _reference_world(json.loads(CONFIG.read_text(encoding="utf-8")))
    public = reference["public"]
    serialized = _canonical(public)
    target = reference["target"]
    assert target["class_id"].encode() not in serialized
    assert target["theorem_id"].encode() not in serialized
    assert target["class_root_sha256"].encode() not in serialized
    assert target["member_root_sha256"].encode() not in serialized
    discovered = _discover(public)
    assert len(discovered) == 1
    assert discovered[0] == target


def test_pre_unseal_leakage_and_actual_read_denial_are_recorded(
    benchmark: dict[str, object],
) -> None:
    pre = benchmark["pre_unseal"]
    audit = pre["leakage_audit"]
    contract = pre["file_read_contract"]
    assert audit["passed"] is True
    assert audit["forbidden_literal_count"] == 0
    assert audit["dependency_paths"] == [
        CONFIG_PATH,
        "src/sigma_theory_compiler/sigma_core.py",
        SOURCE_PATH,
    ]
    assert audit["bytes_scanned"] > 0
    assert contract["attempted_read_count"] == contract["denied_read_count"] == 1
    assert contract["allowed_read_count"] == contract["denied_content_bytes_exposed"] == 0
    assert contract["attempts"] == [
        {"surface": "pathlib.Path.open", "path": TEST_PATH, "decision": "denied"}
    ]
    assert pre["target_identifiers_exposed"] == 0
    assert pre["target_members_exposed"] == 0
    assert pre["target_equivalent_classes_exposed"] == 0


def test_sigma_core_receipts_and_promotion_chain_replay(
    benchmark: dict[str, object],
) -> None:
    candidate = CandidateArtifact.from_dict(benchmark["rediscovery"]["candidate_artifact"])
    stages = [StageOutcome.from_dict(row) for row in benchmark["proof"]["stage_outcomes"]]
    gates = [GateOutcome.from_dict(row) for row in benchmark["proof"]["gate_outcomes"]]
    pre_ledger = PromotionLedger.from_dict(benchmark["proof"]["pre_unseal_ledger"])
    final_ledger = PromotionLedger.from_dict(benchmark["post_unseal"]["final_ledger"])
    assert candidate.content_sha256 == benchmark["rediscovery"]["candidate_root_sha256"]
    assert [row.stage_id for row in stages] == [
        "typed",
        "canonicalized",
        "counterexample_screened",
        "exactly_verified",
    ]
    assert all(row.status.value == "pass" for row in stages)
    assert all(row.status.value == "pass" for row in gates)
    assert pre_ledger.current_stage == "exactly_verified"
    assert final_ledger.current_stage == "prior_art_checked"
    assert pre_ledger.artifact == final_ledger.artifact == candidate.ref
    assert pre_ledger.domain_pack == final_ledger.domain_pack == PACK_DESCRIPTOR.ref


def test_exhaustive_proof_is_sealed_before_post_unseal_comparison(
    benchmark: dict[str, object],
) -> None:
    chronology = benchmark["chronology"]
    assert [row["sequence"] for row in chronology] == list(range(8))
    assert [row["phase"] for row in chronology] == [
        "reference_graph_sealed",
        "public_subgraph_sealed",
        "literal_leakage_audited",
        "discovery_file_reads_denied",
        "missing_cut_class_rediscovered",
        "exhaustive_incidence_proof_sealed",
        "target_unsealed_and_compared",
        "final_promotion_sealed",
    ]
    proof = benchmark["proof"]
    assert proof["method"] == "exhaustive_deterministic_edge_incidence_replay"
    assert proof["canonical_cuts_checked"] == CANONICAL_CUT_COUNT
    assert proof["edge_incidence_tests"] == CANONICAL_CUT_COUNT * EDGE_COUNT == 4080
    assert proof["counterexample_count"] == 0
    assert benchmark["post_unseal"]["comparison_performed_after_proof_seal"] is True
    assert benchmark["post_unseal"]["exact_class_match"] is True


def test_negative_controls_are_executed_and_fail_closed(
    benchmark: dict[str, object],
) -> None:
    controls = {row["control_id"]: row for row in benchmark["negative_controls"]}
    assert set(controls) == {
        "truncated_edge_list_overfit",
        "forbidden_target_file_read",
        "undeclared_graph_lemma_dependency",
        "correctly_typed_unproved_conjecture",
        "resealed_promotion_state",
    }
    overfit = controls["truncated_edge_list_overfit"]
    assert overfit["status"] == "rejected"
    assert overfit["claimed_boundary_size"] != overfit["actual_boundary_size"]
    assert overfit["first_omitted_crossing_edge"] in benchmark["world"]["edges"]
    assert controls["forbidden_target_file_read"] == {
        "control_id": "forbidden_target_file_read",
        "status": "rejected",
        "denied_reads": 1,
        "bytes_exposed": 0,
    }
    dependency = controls["undeclared_graph_lemma_dependency"]
    assert dependency["status"] == "rejected"
    assert dependency["declared_dependency_count"] == 0
    assert len(dependency["blocked_outcome_sha256"]) == 64
    conjecture = controls["correctly_typed_unproved_conjecture"]
    assert conjecture["status"] == "blocked"
    assert conjecture["exact_proof_receipts"] == 0
    assert conjecture["promotion_denied"] is True
    assert len(conjecture["typed_receipt_sha256"]) == 64
    promotion = controls["resealed_promotion_state"]
    assert promotion["status"] == "rejected"
    assert promotion["ledger_validation_rejected"] is True


def test_exact_counts_claims_and_conservative_boundary(benchmark: dict[str, object]) -> None:
    assert benchmark["decision_counts"] == {"pass": 1, "blocked": 0, "reject": 0}
    assert benchmark["metrics"] == {
        "eligible_holdouts": 1,
        "independently_rediscovered_and_proved": 1,
        "proof_rate_numerator": 1,
        "proof_rate_denominator": 1,
        "forbidden_dependency_rejections": 2,
        "counterexample_kills": 1,
        "blocked_unproved_conjectures": 1,
        "forged_promotion_rejections": 1,
    }
    assert benchmark["claims"] == CLAIMS
    assert all(
        benchmark["claims"][key] is False
        for key in (
            "historical_novelty_established",
            "unbounded_graph_discovery_established",
            "general_graph_invariant_completeness_established",
            "formal_proof_assistant_kernel_checked",
            "hostile_process_isolation_established",
            "external_mathematical_significance_established",
        )
    )
    assert benchmark["data_seals"] == EXPECTED_CONFIG["seals"]
    assert "one deterministic anonymous nine-vertex" in benchmark["scope"]


def _mutate_edge(value: dict[str, object]) -> None:
    value["world"]["edges"][0][0] = "n-forged"


def _mutate_graph_count(value: dict[str, object]) -> None:
    value["reference_graph"]["cut_invariant_class_count"] += 1


def _mutate_leakage(value: dict[str, object]) -> None:
    value["pre_unseal"]["leakage_audit"]["passed"] = False


def _mutate_read_count(value: dict[str, object]) -> None:
    value["pre_unseal"]["file_read_contract"]["allowed_read_count"] = 1


def _mutate_candidate(value: dict[str, object]) -> None:
    value["rediscovery"]["candidate_artifact"]["representation"]["cut_size"] += 1


def _mutate_proof(value: dict[str, object]) -> None:
    value["proof"]["edge_incidence_tests"] -= 1


def _mutate_stage_receipt(value: dict[str, object]) -> None:
    value["proof"]["stage_outcomes"][-1]["status"] = "block"


def _mutate_promotion(value: dict[str, object]) -> None:
    value["proof"]["pre_unseal_ledger"]["entries"][-1]["to_stage"] = "prior_art_checked"


def _mutate_unseal_order(value: dict[str, object]) -> None:
    value["chronology"][5], value["chronology"][6] = value["chronology"][6], value["chronology"][5]


def _mutate_target(value: dict[str, object]) -> None:
    value["post_unseal"]["target"]["member_count"] += 1


def _mutate_claim(value: dict[str, object]) -> None:
    value["claims"]["historical_novelty_established"] = True


def _mutate_negative(value: dict[str, object]) -> None:
    value["negative_controls"][0]["status"] = "pass"


def _mutate_unknown(value: dict[str, object]) -> None:
    value["unknown"] = "forged"


def _mutate_source_binding(value: dict[str, object]) -> None:
    value["source_bindings"]["source"]["file_sha256"] = "0" * 64


@pytest.mark.parametrize(
    "mutator",
    [
        _mutate_edge,
        _mutate_graph_count,
        _mutate_leakage,
        _mutate_read_count,
        _mutate_candidate,
        _mutate_proof,
        _mutate_stage_receipt,
        _mutate_promotion,
        _mutate_unseal_order,
        _mutate_target,
        _mutate_claim,
        _mutate_negative,
        _mutate_unknown,
        _mutate_source_binding,
    ],
)
def test_resealed_semantic_and_provenance_tampering_fails_closed(
    benchmark: dict[str, object], mutator: Callable[[dict[str, object]], None]
) -> None:
    forged = deepcopy(benchmark)
    mutator(forged)
    forged = _reseal(forged)
    with pytest.raises((ValueError, TypeError)):
        validate_result(forged, root=ROOT)


def test_config_and_path_boundaries_fail_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["world_generator"]["vertex_count"] += 1
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(config)
    with pytest.raises(ValueError, match="escapes project root"):
        _inside(ROOT, "../outside.json")
    with pytest.raises(ValueError, match="registered config path"):
        build_benchmark(ROOT / "configs" / "not_registered.json")


def test_all_four_local_bindings_are_live(benchmark: dict[str, object]) -> None:
    expected = {
        "config": CONFIG_PATH,
        "core": "src/sigma_theory_compiler/sigma_core.py",
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    assert set(benchmark["source_bindings"]) == set(expected)
    for label, relative in expected.items():
        raw = (ROOT / relative).read_bytes()
        assert benchmark["source_bindings"][label] == {
            "path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
        }
