from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_p10_inverse_product_d2_replay_gate import (
    CONFIG_PATH,
    D1_DAG_SHA256,
    FIRST_BLOCKER,
    LEAF_DAG_SHA256,
    OUTPUT_PATH,
    _children,
    _content_sha,
    _dense_leaf_roots,
    _load_D1,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return build_gate(ROOT / CONFIG_PATH)


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value["content_sha256"] = _content_sha(value)
    return value


def test_build_matches_immutable_artifact(result: dict[str, object]) -> None:
    stored = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    assert result == stored
    assert stored["content_sha256"] == _content_sha(stored)


def test_exact_candidate_and_root_census(result: dict[str, object]) -> None:
    assert result["decision"] == (
        "pass_all_84_P10_ordered_D2_roots_exactly_replayed_Pother_blocked"
    )
    assert result["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert result["downstream_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }
    counts = result["gate_counts"]
    assert counts["registered_P10_leaf_derivative_roots_consumed"] == 7920
    assert counts["unique_P10_replay_roots"] == 60
    assert counts["sealed_P10_ordered_D2_roots"] == 84
    assert counts["blocked_P10_ordered_D2_roots"] == 0
    assert counts["Pother_ordered_D2_roots_registered"] == 0
    assert counts["Pother_ordered_D2_roots_blocked"] == 180
    assert counts["all_target_ordered_D2_roots_registered"] == 84
    assert counts["all_target_ordered_D2_roots_blocked"] == 180


def test_every_candidate_has_five_replays_and_seven_ordered_records(
    result: dict[str, object],
) -> None:
    manifests = result["candidate_manifests"]
    assert len(manifests) == 12
    record_ids: set[str] = set()
    replay_roots: set[str] = set()
    for manifest in manifests:
        assert manifest["unique_P10_replay_roots"] == 5
        assert manifest["sealed_ordered_P10_D2_roots"] == 7
        assert manifest["blocked_ordered_Pother_D2_roots"] == 15
        assert manifest["candidate_rejection_authorized"] is False
        assert manifest["first_blocker"] == FIRST_BLOCKER
        assert len(manifest["replay_packets"]) == 5
        assert len(manifest["ordered_P10_D2_records"]) == 7
        for packet in manifest["replay_packets"]:
            assert packet["D1_arithmetic_dag_sha256"] == D1_DAG_SHA256
            assert packet["leaf_derivative_arithmetic_dag_sha256"] == LEAF_DAG_SHA256
            assert packet["bound_leaf_derivative_count"] == 132
            assert packet["nonzero_bound_leaf_derivative_count"] == 4
            assert packet["D2_merkle_replay_node_count"] > 0
            assert packet["quotient_domain_assumption"] == ("c11=(-1)^11 det(A) is nonzero")
            replay_roots.add(packet["D2_merkle_replay_root_sha256"])
        for record in manifest["ordered_P10_D2_records"]:
            assert record["root_status"] == ("sealed_exact_arbitrary_background_merkle_replay")
            assert record["candidate_rejection_authorized"] is False
            assert record["ordered_D2_record_id"] not in record_ids
            record_ids.add(record["ordered_D2_record_id"])
    assert len(record_ids) == 84
    assert len(replay_roots) == 60


def test_replay_operation_contract_is_closed(result: dict[str, object]) -> None:
    contract = result["replay_contract"]
    assert contract["bound_D1_arithmetic_dag_sha256"] == D1_DAG_SHA256
    assert contract["bound_leaf_derivative_arithmetic_dag_sha256"] == LEAF_DAG_SHA256
    assert len(contract["closed_derivative_rules"]) == 6
    assert contract["full_trace_recomputed_during_validation"] is True
    allowed = {
        "exact_constant",
        "bound_D1_primal_node_reference",
        "bound_leaf_derivative_root_reference",
        "exact_add",
        "exact_negate",
        "exact_multiply",
        "exact_divide",
    }
    for manifest in result["candidate_manifests"]:
        for packet in manifest["replay_packets"]:
            assert set(packet["D2_merkle_replay_operation_counts"]) <= allowed
            assert packet["D2_merkle_replay_operation_counts"]["exact_divide"] > 0


def test_claims_remain_fail_closed(result: dict[str, object]) -> None:
    seals = result["claim_seals"]
    assert seals["all_84_P10_ordered_D2_roots_exactly_replayed"] is True
    for key in (
        "Pother_leaf_derivative_roots_registered",
        "Pother_ordered_D2_roots_registered",
        "physical_no_go_proved",
        "complete_ordered_D2F_tensor_registered",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "candidate_theory_rejected",
        "observational_claim_made",
    ):
        assert seals[key] is False
    assert set(result["data_seals"].values()) == {False}
    assert all(row == {"rejected": True} for row in result["exact_controls"].values())


def test_source_config_test_and_inputs_are_hash_bound(result: dict[str, object]) -> None:
    bindings = result["source_bindings"]
    assert set(bindings) == {
        "source",
        "config",
        "test",
        "predecessor",
        "direct_D1_artifact",
    }
    for label in ("source", "config", "test"):
        path = ROOT / bindings[label]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == bindings[label]["file_sha256"]
    for bundle in (bindings["predecessor"],):
        for label in ("source", "config", "test", "artifact"):
            path = ROOT / bundle[label]["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == bundle[label]["file_sha256"]
    d1 = bindings["direct_D1_artifact"]
    assert hashlib.sha256((ROOT / d1["path"]).read_bytes()).hexdigest() == d1["file_sha256"]


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("decision", "pass_complete_D2F"),
        ("first_blocker", "none"),
        ("manifest_sha256", "0" * 64),
        ("scope", "global theorem"),
    ],
)
def test_resealed_top_level_tamper_fails_closed(
    result: dict[str, object], key: str, replacement: object
) -> None:
    tampered = copy.deepcopy(result)
    tampered[key] = replacement
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("gate_counts", "sealed_P10_ordered_D2_roots", 83),
        ("gate_counts", "Pother_ordered_D2_roots_registered", 1),
        ("claim_seals", "complete_ordered_D2F_tensor_registered", True),
        ("claim_seals", "physical_no_go_proved", True),
        ("data_seals", "live_SQLite_opened", True),
        ("exact_controls", "default_unregistered_leaf_derivative_to_zero", {"rejected": False}),
    ],
)
def test_resealed_nested_tamper_fails_closed(
    result: dict[str, object], section: str, key: str, replacement: object
) -> None:
    tampered = copy.deepcopy(result)
    tampered[section][key] = replacement
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_resealed_replay_root_tamper_fails_closed(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    tampered["candidate_manifests"][0]["replay_packets"][0]["D2_merkle_replay_root_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_resealed_trace_hash_tamper_fails_closed(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    tampered["candidate_manifests"][0]["replay_packets"][0]["D2_merkle_replay_trace_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_resealed_ordered_record_tamper_fails_closed(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    tampered["candidate_manifests"][0]["ordered_P10_D2_records"][0]["coordinate_ordinal"] = 99
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_unknown_D1_operator_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown D1 operator"):
        _children({"op": "symbolic_differentiate"})


def test_malformed_D1_manifest_is_rejected() -> None:
    d1 = json.loads(
        (
            ROOT
            / "runs/physics-language/quartic-full-source-jacobian-arithmetic-campaign/campaign.json"
        ).read_text(encoding="utf-8")
    )
    d1["common_full_entry_manifest"]["entries"].pop()
    with pytest.raises(ValueError, match="D1 manifest changed"):
        _load_D1(d1)


def test_unknown_D1_allowed_operation_is_rejected() -> None:
    d1 = json.loads(
        (
            ROOT
            / "runs/physics-language/quartic-full-source-jacobian-arithmetic-campaign/campaign.json"
        ).read_text(encoding="utf-8")
    )
    d1["common_principal_arithmetic_packet"]["arithmetic_dag"]["allowed_operations"].append(
        "differentiate"
    )
    with pytest.raises(ValueError, match="D1 DAG changed"):
        _load_D1(d1)


def test_incomplete_leaf_packet_is_rejected() -> None:
    predecessor = json.loads(
        (
            ROOT
            / "runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json"
        ).read_text(encoding="utf-8")
    )
    packet = copy.deepcopy(predecessor["candidate_manifests"][0]["direction_packets"][0])
    packet["A_derivative_shape"] = [10, 11]
    with pytest.raises(ValueError, match="leaf packet changed"):
        _dense_leaf_roots(packet)


def test_unknown_top_level_key_fails_closed(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    tampered["unregistered_claim"] = True
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)
