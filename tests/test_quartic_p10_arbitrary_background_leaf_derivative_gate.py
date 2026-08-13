from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_p10_arbitrary_background_leaf_derivative_gate import (
    CONFIG_PATH,
    EXPECTED_EVIDENCE,
    EXPECTED_PREDECESSOR,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    SOURCE_PATH,
    TARGET_SYMBOLS,
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
    with pytest.raises(ValueError, match="P10 arbitrary-background leaf"):
        _validate_result(value, root=ROOT)


def _packets(artifact: dict) -> list[dict]:
    return [
        packet
        for manifest in artifact["candidate_manifests"]
        for packet in manifest["direction_packets"]
    ]


def test_checked_artifact_is_exact_rebuild() -> None:
    artifact = _load()
    assert build_gate(ROOT / CONFIG_PATH) == artifact
    _validate_result(artifact, root=ROOT)


def test_nonlinear_geometric_map_is_directly_bound() -> None:
    binding = _load()["nonlinear_geometric_map_binding"]
    assert binding["formula_contract_sha256"] == (
        "9d0a41e02f3a86b4f6351240d57078e859dd9b6ce047bcaf1b08b71e2296cb11"
    )
    assert binding["status"] == "pass_all_12_exact_nonlinear_geometric_state_to_jet_maps"
    assert binding["arbitrary_background_scope"] is True


def test_exact_five_P10_covariant_tangents_are_registered() -> None:
    packets = _load()["generic_derivative_packets"]
    assert len(packets) == 5
    assert {row["coordinate_atom"] for row in packets} == set(TARGET_SYMBOLS)
    for row in packets:
        assert row["covariant_tangent"] == {TARGET_SYMBOLS[row["coordinate_atom"]]: "1"}
        assert row["covariant_tangent_background_independent"] is True
    assert sum(len(row["coordinate_ordinals"]) for row in packets) == 7


def test_generic_leaf_derivative_census_is_exact() -> None:
    packets = _load()["generic_derivative_packets"]
    assert all(row["total_leaf_derivatives"] == 132 for row in packets)
    assert sum(row["nonzero_leaf_derivatives"] for row in packets) == 20
    assert sum(row["zero_leaf_derivatives"] for row in packets) == 640
    assert all(row["source_chunk_input_column"] == 10 for row in packets)


def test_all_candidate_leaf_roots_are_compactly_registered() -> None:
    artifact = _load()
    packets = _packets(artifact)
    assert len(artifact["candidate_manifests"]) == 12
    assert len(packets) == 60
    assert all(packet["total_leaf_derivative_roots"] == 132 for packet in packets)
    assert sum(packet["total_leaf_derivative_roots"] for packet in packets) == 7920
    assert sum(packet["nonzero_leaf_derivative_roots"] for packet in packets) == 240
    assert all(packet["arbitrary_background_valid"] is True for packet in packets)


def test_sparse_plus_zero_default_replays_dense_root_manifests() -> None:
    artifact = _load()
    dag = artifact["leaf_derivative_arithmetic_DAG"]
    assert dag["allowed_operations"] == ["exact_constant"]
    assert dag["node_count"] > 1
    for packet in _packets(artifact):
        assert 0 <= packet["zero_default_arithmetic_root"] < dag["node_count"]
        for entry in [
            *packet["A_derivative_sparse_entries"],
            *packet["source_chunk_column_derivative_sparse_entries"],
        ]:
            assert 0 <= entry["arithmetic_root"] < dag["node_count"]
            assert dag["nodes"][entry["arithmetic_root"]]["value"] == entry["value"]
        assert len(packet["dense_root_manifest_sha256"]) == 64


def test_each_candidate_counts_are_exact_and_fail_closed() -> None:
    for manifest in _load()["candidate_manifests"]:
        assert manifest["unique_P10_directions"] == 5
        assert manifest["P10_target_records"] == 7
        assert manifest["registered_leaf_derivative_roots"] == 660
        assert manifest["nonzero_leaf_derivative_roots"] == 20
        assert manifest["zero_leaf_derivative_roots"] == 640
        assert manifest["P10_ordered_D2_roots_registered"] == 0
        assert manifest["candidate_rejection_authorized"] is False


def test_global_counts_preserve_remaining_boundary() -> None:
    artifact = _load()
    counts = artifact["gate_counts"]
    assert counts["registered_arbitrary_background_leaf_derivative_roots"] == 7920
    assert counts["nonzero_leaf_derivative_roots"] == 240
    assert counts["zero_leaf_derivative_roots"] == 7680
    assert counts["P10_ordered_D2_roots_registered"] == 0
    assert counts["P10_ordered_D2_roots_blocked"] == 84
    assert counts["Pother_leaf_derivative_roots_remaining"] == 23760
    assert counts["all_target_ordered_D2_roots_blocked"] == 264
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_decision_counts_do_not_promote_D2_or_reject() -> None:
    artifact = _load()
    assert artifact["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert artifact["downstream_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }


def test_claims_distinguish_leaf_registration_from_D2() -> None:
    claims = _load()["claim_seals"]
    assert claims["nonlinear_geometric_map_directly_bound"] is True
    assert claims["P10_coordinate_to_Hessian_tangents_arbitrary_background_registered"] is True
    assert claims["all_7920_P10_leaf_derivative_roots_registered"] is True
    assert claims["P10_ordered_D2_roots_registered"] is False
    assert claims["Pother_leaf_derivative_roots_registered"] is False
    assert claims["physical_no_go_proved"] is False


def test_all_broad_claims_remain_closed() -> None:
    claims = _load()["claim_seals"]
    for key in (
        "complete_ordered_D2F_tensor_registered",
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
    assert bindings["direct_evidence"] == EXPECTED_EVIDENCE


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
        ("gate_counts", "registered_arbitrary_background_leaf_derivative_roots", 7919),
        ("gate_counts", "P10_ordered_D2_roots_registered", 1),
        ("gate_counts", "Pother_leaf_derivative_roots_registered", 1),
        ("claim_seals", "P10_ordered_D2_roots_registered", True),
        ("claim_seals", "Pother_leaf_derivative_roots_registered", True),
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
        ("coordinate_atom", "unknown"),
        ("total_leaf_derivative_roots", 131),
        ("nonzero_leaf_derivative_roots", 99),
        ("zero_default_arithmetic_root", 999),
        ("arbitrary_background_valid", False),
    ],
)
def test_resealed_direction_packet_tampers_are_rejected(field: str, replacement: object) -> None:
    value = _load()
    value["candidate_manifests"][0]["direction_packets"][0][field] = replacement
    _reject(value)


def test_resealed_sparse_leaf_root_tamper_is_rejected() -> None:
    value = _load()
    value["candidate_manifests"][0]["direction_packets"][0]["A_derivative_sparse_entries"][0][
        "arithmetic_root"
    ] = 999
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


@pytest.mark.parametrize("bundle", sorted(EXPECTED_EVIDENCE))
def test_resealed_evidence_binding_tampers_are_rejected(bundle: str) -> None:
    value = _load()
    value["source_bindings"]["direct_evidence"][bundle]["artifact"]["content_sha256"] = "0" * 64
    _reject(value)
