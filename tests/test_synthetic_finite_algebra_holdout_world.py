from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.sigma_core import (
    CandidateArtifact,
    GateOutcome,
    PromotionLedger,
    StageOutcome,
)
from sigma_theory_compiler.synthetic_finite_algebra_holdout_world import (
    ASSIGNMENT_COUNT,
    CLAIMS,
    CONFIG_PATH,
    EXPECTED_CONFIG,
    OUTPUT_PATH,
    PACK_DESCRIPTOR,
    SOURCE_PATH,
    TEST_PATH,
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
    assert (
        checked["content_sha256"]
        == hashlib.sha256(
            _canonical({key: item for key, item in checked.items() if key != "content_sha256"})
        ).hexdigest()
    )


def test_fresh_world_axioms_and_reference_graph_are_exact(
    benchmark: dict[str, object],
) -> None:
    world = benchmark["world"]
    graph = benchmark["reference_graph"]
    assert world["generation_epoch"] == "2026-08-12"
    assert world["order"] == 7
    assert len(world["operation_table"]) == 7
    assert all(len(row) == 7 and sorted(row) == list(range(7)) for row in world["operation_table"])
    assert all(
        sorted(world["operation_table"][row][column] for row in range(7)) == list(range(7))
        for column in range(7)
    )
    assert graph["axiom_count"] == 1
    assert graph["visible_ancestor_count"] == 2
    assert graph["hidden_theorem_class_count"] == 1
    assert graph["visible_theorem_class_count"] + 1 == graph["nontrivial_theorem_class_count"]
    assert {row["theorem_id"] for row in graph["visible_ancestors"]} == {
        "ancestor.left_translations_bijective",
        "ancestor.right_translations_bijective",
    }
    assert all(row["cases_checked"] == 49 for row in graph["visible_ancestors"])


def test_complete_term_enumeration_and_hidden_class_rediscovery(
    benchmark: dict[str, object],
) -> None:
    pre = benchmark["pre_unseal"]
    rediscovery = benchmark["rediscovery"]
    assert pre["raw_term_count"] == 24
    assert pre["assignments_per_term"] == ASSIGNMENT_COUNT == 2401
    assert pre["term_evaluations"] == 24 * 2401
    assert pre["target_identifiers_exposed"] == 0
    assert pre["target_equations_exposed"] == 0
    assert pre["target_equivalent_classes_exposed"] == 0
    assert rediscovery["candidate_class_count"] == 1
    assert rediscovery["answer_bearing_dependencies_used"] == 0
    assert benchmark["post_unseal"]["exact_class_match"] is True
    assert (
        rediscovery["candidate_artifact"]["representation"]["class_id"]
        == benchmark["post_unseal"]["target"]["class_id"]
    )


def test_discovery_function_receives_public_subgraph_not_target() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    reference = _reference_world(config)
    public_bytes = _canonical(reference["public"])
    target = reference["target"]
    assert target["class_id"].encode() not in public_bytes
    assert target["theorem_id"].encode() not in public_bytes
    assert target["class_root_sha256"].encode() not in public_bytes
    assert target["semantic_vector_sha256"].encode() not in public_bytes
    discovered = _discover(reference["public"])
    assert len(discovered) == 1
    assert discovered[0]["class_id"] == target["class_id"]


def test_pre_unseal_leakage_and_actual_read_denial_are_recorded(
    benchmark: dict[str, object],
) -> None:
    pre = benchmark["pre_unseal"]
    leakage = pre["leakage_audit"]
    reads = pre["file_read_contract"]
    assert leakage["dependency_paths"] == [
        CONFIG_PATH,
        "src/sigma_theory_compiler/sigma_core.py",
        SOURCE_PATH,
    ]
    assert leakage["forbidden_literal_count"] == 0
    assert leakage["forbidden_literal_labels_found"] == []
    assert leakage["passed"] is True
    assert reads["attempted_read_count"] == 1
    assert reads["allowed_read_count"] == 0
    assert reads["denied_read_count"] == 1
    assert reads["denied_content_bytes_exposed"] == 0
    assert reads["attempts"] == [
        {
            "surface": "pathlib.Path.open",
            "path": TEST_PATH,
            "decision": "denied",
        }
    ]
    assert "not_an_operating_system_sandbox" in reads["enforcement_scope"]


def test_sigma_core_candidate_receipts_and_promotion_chain_replay(
    benchmark: dict[str, object],
) -> None:
    candidate = CandidateArtifact.from_dict(benchmark["rediscovery"]["candidate_artifact"])
    stages = [StageOutcome.from_dict(row) for row in benchmark["proof"]["stage_outcomes"]]
    gates = [GateOutcome.from_dict(row) for row in benchmark["proof"]["gate_outcomes"]]
    pre_ledger = PromotionLedger.from_dict(benchmark["proof"]["pre_unseal_ledger"])
    comparison = StageOutcome.from_dict(benchmark["post_unseal"]["comparison_outcome"])
    comparison_gate = GateOutcome.from_dict(benchmark["post_unseal"]["comparison_gate"])
    final_ledger = PromotionLedger.from_dict(benchmark["post_unseal"]["final_ledger"])
    assert candidate.provenance.domain_pack == PACK_DESCRIPTOR.ref
    assert [row.stage_id for row in stages] == [
        "typed",
        "canonicalized",
        "counterexample_screened",
        "exactly_verified",
    ]
    assert all(row.status.value == "pass" for row in stages)
    assert all(row.status.value == "pass" for row in gates)
    assert pre_ledger.current_stage == "exactly_verified"
    assert comparison.stage_id == "prior_art_checked"
    assert comparison_gate.gate_id == "admit_prior_art_checked"
    assert final_ledger.current_stage == "prior_art_checked"
    assert len(pre_ledger.entries) == 4
    assert len(final_ledger.entries) == 5


def test_exhaustive_proof_is_sealed_before_post_unseal_comparison(
    benchmark: dict[str, object],
) -> None:
    proof = benchmark["proof"]
    chronology = benchmark["chronology"]
    assert proof["method"] == "exhaustive_deterministic_semantics"
    assert proof["assignments_checked"] == 2401
    assert proof["counterexample_count"] == 0
    assert (
        proof["proof_seal"]["candidate_artifact_sha256"]
        == benchmark["rediscovery"]["candidate_root_sha256"]
    )
    phases = [row["phase"] for row in chronology]
    assert phases == [
        "reference_graph_sealed",
        "public_subgraph_sealed",
        "literal_leakage_audited",
        "discovery_file_reads_denied",
        "missing_class_rediscovered",
        "exhaustive_proof_sealed",
        "target_unsealed_and_compared",
        "final_promotion_sealed",
    ]
    assert phases.index("exhaustive_proof_sealed") < phases.index("target_unsealed_and_compared")
    assert [row["sequence"] for row in chronology] == list(range(8))


def test_negative_controls_are_exact_and_fail_closed(benchmark: dict[str, object]) -> None:
    controls = {row["control_id"]: row for row in benchmark["negative_controls"]}
    assert set(controls) == {
        "example_prefix_overfit",
        "forbidden_target_file_read",
        "undeclared_lemma_dependency",
        "correctly_typed_unproved_conjecture",
        "resealed_promotion_state",
    }
    assert controls["example_prefix_overfit"]["status"] == "rejected"
    assert controls["example_prefix_overfit"]["visible_prefix_points_passed"] >= 1
    assert controls["example_prefix_overfit"]["first_counterexample"] is not None
    assert controls["forbidden_target_file_read"]["denied_reads"] == 1
    assert controls["forbidden_target_file_read"]["bytes_exposed"] == 0
    assert controls["undeclared_lemma_dependency"]["status"] == "rejected"
    assert len(controls["undeclared_lemma_dependency"]["blocked_outcome_sha256"]) == 64
    assert controls["correctly_typed_unproved_conjecture"]["status"] == "blocked"
    assert controls["correctly_typed_unproved_conjecture"]["promotion_denied"] is True
    assert len(controls["correctly_typed_unproved_conjecture"]["typed_receipt_sha256"]) == 64
    assert controls["resealed_promotion_state"]["status"] == "rejected"
    assert controls["resealed_promotion_state"]["ledger_validation_rejected"] is True


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
    assert {key for key, value in CLAIMS.items() if value} == {
        "fresh_seed_derived_anonymous_finite_world_generated",
        "complete_declared_term_grammar_enumerated",
        "reference_theorem_graph_sealed_before_discovery",
        "entire_target_equivalence_class_withheld",
        "visible_ancestor_theorems_exposed",
        "pre_unseal_answer_literal_leakage_absent",
        "withheld_class_independently_rediscovered",
        "exhaustive_finite_semantics_proof_completed",
        "post_unseal_equivalence_confirmed",
    }
    assert "one deterministic anonymous order-seven" in benchmark["scope"]
    assert "no historical novelty" in benchmark["scope"]
    assert not any(benchmark["data_seals"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("table", "result boundary"),
        ("graph_count", "result boundary"),
        ("leakage", "result boundary"),
        ("read_count", "result boundary"),
        ("candidate", "result boundary"),
        ("proof", "result boundary"),
        ("stage_receipt", "result boundary"),
        ("promotion", "result boundary"),
        ("unseal_order", "result boundary"),
        ("target", "result boundary"),
        ("claim", "result boundary"),
        ("negative", "result boundary"),
        ("unknown", "result boundary"),
        ("source_binding", "source binding"),
    ],
)
def test_resealed_semantic_and_provenance_tampering_fails_closed(
    benchmark: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(benchmark))
    if mutation == "table":
        value["world"]["operation_table"][0][0] = 99
    elif mutation == "graph_count":
        value["reference_graph"]["hidden_theorem_class_count"] = 0
    elif mutation == "leakage":
        value["pre_unseal"]["leakage_audit"]["forbidden_literal_count"] = 1
    elif mutation == "read_count":
        value["pre_unseal"]["file_read_contract"]["denied_read_count"] = 0
    elif mutation == "candidate":
        value["rediscovery"]["candidate_artifact"]["statement"] = "forged"
    elif mutation == "proof":
        value["proof"]["assignments_checked"] = 2400
    elif mutation == "stage_receipt":
        value["proof"]["stage_outcomes"][3]["outcome_sha256"] = "0" * 64
    elif mutation == "promotion":
        value["post_unseal"]["final_ledger"]["ledger_sha256"] = "0" * 64
    elif mutation == "unseal_order":
        value["chronology"][5]["phase"] = "target_unsealed_and_compared"
    elif mutation == "target":
        value["post_unseal"]["target"]["class_id"] = "eqc-forged"
    elif mutation == "claim":
        value["claims"]["historical_novelty_established"] = True
    elif mutation == "negative":
        value["negative_controls"][0]["status"] = "passed"
    elif mutation == "unknown":
        value["benchmark_wide_success"] = True
    else:
        value["source_bindings"]["source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        validate_result(_reseal(value), root=ROOT)


def test_config_and_path_boundaries_fail_closed() -> None:
    config = json.loads(json.dumps(EXPECTED_CONFIG))
    config["policies"]["target_access_before_proof_seal"] = "allowed"
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(config)
    with pytest.raises(ValueError, match="path escapes"):
        _inside(ROOT, "../outside.json")


def test_all_four_local_bindings_are_live(benchmark: dict[str, object]) -> None:
    expected = {
        "config": CONFIG_PATH,
        "core": "src/sigma_theory_compiler/sigma_core.py",
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    assert set(benchmark["source_bindings"]) == set(expected)
    for label, relative in expected.items():
        assert benchmark["source_bindings"][label] == {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
