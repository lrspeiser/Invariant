from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_ordered_mixed_d2_arithmetic_dag_differentiability_gate import (
    CONFIG_PATH,
    DAG_SHA256,
    EXPECTED_PREDECESSOR,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / OUTPUT_PATH


def _load() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reject(value: dict) -> None:
    _reseal(value)
    with pytest.raises(ValueError, match="D2 DAG differentiability"):
        _validate_result(value, root=ROOT)


def _packets(artifact: dict) -> list[dict]:
    return [
        packet
        for manifest in artifact["candidate_manifests"]
        for packet in manifest["derivative_packets"]
    ]


def _obligations(artifact: dict) -> list[dict]:
    return [row for packet in _packets(artifact) for row in packet["leaf_derivative_obligations"]]


def test_checked_artifact_is_exact_rebuild() -> None:
    artifact = _load()
    assert build_gate(ROOT / CONFIG_PATH) == artifact
    _validate_result(artifact, root=ROOT)


def test_target_root_templates_are_exact_and_deduplicated() -> None:
    templates = _load()["target_root_templates"]
    assert len(templates) == 20
    assert len({row["D1_arithmetic_root"] for row in templates}) == 20
    assert all(row["D1_arithmetic_dag_sha256"] == DAG_SHA256 for row in templates)
    assert all(row["reachable_component_input_count"] == 132 for row in templates)
    assert sum(len(row["coordinate_ordinals"]) for row in templates) == 22


def test_union_dependency_closure_is_exact() -> None:
    union = _load()["union_dependency_closure"]
    assert union["reachable_nodes"] == 13983
    assert union["reachable_component_input_labels"] == 341
    assert union["operation_counts"] == {
        "exact_add": 1241,
        "exact_component_input": 341,
        "exact_constant": 11,
        "exact_divide": 22,
        "exact_multiply": 12326,
        "exact_negate": 42,
    }
    assert sum(union["component_input_family_counts"].values()) == 341


def test_operator_rules_are_closed_except_for_input_leaf_jets() -> None:
    rules = _load()["operator_derivative_rules"]
    assert rules["closed_noninput_operator_rules"] == 5
    assert rules["unbound_input_operator_kinds"] == 1
    assert rules["exact_component_input"] == (
        "requires_registered_candidate_coordinate_derivative_leaf"
    )
    assert rules["derivative_DAG_emitted"] is False


def test_all_minimal_leaf_obligations_are_materialized_and_unique() -> None:
    artifact = _load()
    obligations = _obligations(artifact)
    assert len(obligations) == 31680
    assert len({row["leaf_derivative_obligation_id"] for row in obligations}) == 31680
    assert len(_packets(artifact)) == 240
    assert all(packet["required_leaf_derivatives"] == 132 for packet in _packets(artifact))


def test_each_candidate_has_2640_deduplicated_leaf_obligations() -> None:
    for manifest in _load()["candidate_manifests"]:
        assert manifest["unique_target_D1_roots"] == 20
        assert manifest["required_leaf_derivative_obligations"] == 2640
        assert manifest["registered_leaf_derivative_roots"] == 0
        assert manifest["ordered_mixed_D2_roots_registered"] == 0
        assert len(manifest["derivative_packets"]) == 20


def test_leaf_obligations_are_bound_but_values_are_absent() -> None:
    for row in _obligations(_load()):
        assert row["candidate_id"]
        assert row["target_coordinate_atom"]
        assert row["component_input_label"]
        assert len(row["component_input_provenance_sha256"]) == 64
        assert len(row["tangent_embedding_id"]) == 64
        assert row["component_input_coordinate_derivative_root_registered"] is False
        assert row["component_input_coordinate_derivative_dag_sha256_registered"] is False
        assert row["obligation_status"] == "required_unregistered"
        assert row["candidate_rejection_authorized"] is False


def test_counts_preserve_fail_closed_boundary() -> None:
    artifact = _load()
    counts = artifact["gate_counts"]
    assert counts["target_coordinate_records"] == 264
    assert counts["raw_leaf_derivative_references"] == 34848
    assert counts["deduplicated_leaf_derivative_obligations"] == 31680
    assert counts["registered_leaf_derivative_roots"] == 0
    assert counts["registered_ordered_mixed_D2_roots"] == 0
    assert counts["blocked_ordered_mixed_D2_roots"] == 264
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_decision_counts_do_not_reject_or_promote() -> None:
    artifact = _load()
    assert artifact["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert artifact["downstream_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }


def test_claims_distinguish_schema_obstruction_from_physical_no_go() -> None:
    claims = _load()["claim_seals"]
    assert claims["target_D1_DAG_dependency_closure_replayed"] is True
    assert claims["exact_noninput_operator_derivative_rules_closed"] is True
    assert claims["minimal_31680_leaf_derivative_obligations_materialized"] is True
    assert claims["component_input_leaf_derivatives_registered"] is False
    assert claims["ordered_mixed_D2_values_registered"] is False
    assert claims["physical_no_go_proved"] is False


def test_all_broad_claims_remain_closed() -> None:
    claims = _load()["claim_seals"]
    for key in (
        "complete_ordered_D2F_tensor_registered",
        "full_high_atom_good_unknown_identity_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "candidate_theory_rejected",
        "observational_claim_made",
    ):
        assert claims[key] is False


def test_live_bindings_are_exact() -> None:
    bindings = _load()["source_bindings"]
    assert bindings["source"] == {"path": SOURCE_PATH, "file_sha256": _file_sha(ROOT / SOURCE_PATH)}
    assert bindings["config"] == {"path": CONFIG_PATH, "file_sha256": _file_sha(ROOT / CONFIG_PATH)}
    assert bindings["test"] == {"path": TEST_PATH, "file_sha256": _file_sha(ROOT / TEST_PATH)}
    assert bindings["predecessor"] == EXPECTED_PREDECESSOR
    assert bindings["full_source_D1_artifact"]["content_sha256"] == (
        "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a"
    )


def test_raw_content_tamper_is_rejected() -> None:
    value = _load()
    value["scope"] = "tampered"
    with pytest.raises(ValueError, match="content hash"):
        _validate_result(value, root=ROOT)


def test_unknown_top_level_key_is_rejected() -> None:
    value = _load()
    value["unknown"] = True
    _reject(value)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("decision_counts", "reject", 1),
        ("downstream_admission_counts", "pass", 1),
        ("gate_counts", "deduplicated_leaf_derivative_obligations", 31679),
        ("gate_counts", "registered_leaf_derivative_roots", 1),
        ("gate_counts", "registered_ordered_mixed_D2_roots", 1),
        ("operator_derivative_rules", "derivative_DAG_emitted", True),
        ("claim_seals", "component_input_leaf_derivatives_registered", True),
        ("claim_seals", "ordered_mixed_D2_values_registered", True),
        ("claim_seals", "physical_no_go_proved", True),
        ("claim_seals", "candidate_theory_rejected", True),
    ],
)
def test_resealed_summary_tampers_are_rejected(
    section: str, field: str, replacement: object
) -> None:
    value = _load()
    value[section][field] = replacement
    _reject(value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("leaf_derivative_obligation_id", "0" * 64),
        ("candidate_id", "unknown"),
        ("component_input_label", "unknown"),
        ("target_D1_arithmetic_root", 0),
        ("component_input_coordinate_derivative_root_registered", True),
        ("component_input_coordinate_derivative_dag_sha256_registered", True),
        ("candidate_rejection_authorized", True),
    ],
)
def test_resealed_leaf_obligation_tampers_are_rejected(field: str, replacement: object) -> None:
    value = _load()
    value["candidate_manifests"][0]["derivative_packets"][0]["leaf_derivative_obligations"][0][
        field
    ] = replacement
    _reject(value)


@pytest.mark.parametrize("binding", ["source", "config", "test"])
def test_resealed_local_binding_tampers_are_rejected(binding: str) -> None:
    value = _load()
    value["source_bindings"][binding]["file_sha256"] = "0" * 64
    _reject(value)


@pytest.mark.parametrize("binding", ["source", "config", "test", "artifact"])
def test_resealed_predecessor_binding_tampers_are_rejected(binding: str) -> None:
    value = _load()
    value["source_bindings"]["predecessor"][binding]["file_sha256"] = "0" * 64
    _reject(value)


def test_resealed_full_source_binding_tamper_is_rejected() -> None:
    value = _load()
    value["source_bindings"]["full_source_D1_artifact"]["content_sha256"] = "0" * 64
    _reject(value)
