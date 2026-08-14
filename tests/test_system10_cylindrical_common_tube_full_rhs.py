from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_common_tube_full_rhs import (
    DECISION,
    System10CommonTubeFullRHSError,
    _verify_candidate_packet,
    build_candidate_packet,
    build_receipt,
)
from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _load_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_common_tube_full_rhs.json"
OUTPUT = ROOT / "runs/math/system10-cylindrical-common-tube-full-rhs"


@pytest.fixture(scope="module")
def config() -> dict:
    return _load_json(CONFIG)


@pytest.fixture(scope="module")
def packet() -> dict:
    return build_candidate_packet(CONFIG, 0, root=ROOT)


def test_candidate_packet_closes_exact_partition_and_origins(packet: dict) -> None:
    assert len(packet["base_74_row_references"]) == 74
    assert len(packet["dynamic_11_rows"]) == 11
    assert packet["row_partition"]["complete_0_through_84"] is True
    assert sorted(packet["row_partition"]["lhs_state_indices"]) == list(range(85))
    assert all(row["exact_residual_replay"]["zero"] for row in packet["dynamic_11_rows"])
    assert all(row["equation_origin"]["origin_sha256"] for row in packet["dynamic_11_rows"])
    assert packet["claims"]["candidate_common_tube_exact_85_state_rhs_closed"] is True
    assert packet["claims"]["fixed_r_positive_domain_full_rhs_closed"] is False
    assert packet["claims"]["constraint_propagation_closed"] is False
    assert packet["claims"]["hyperbolicity_closed"] is False


def test_linked_rhs_nodes_are_explicitly_bound(packet: dict) -> None:
    assert any(len(row["rhs_W_nodes"]) > 1 for row in packet["dynamic_11_rows"])
    for row in packet["dynamic_11_rows"]:
        assert row["rhs_representation"] == "sealed_linked_exact_dag"
        assert row["rhs_acceleration_expression"]
        for node in row["rhs_W_nodes"]:
            assert node["source_W_entry_sha256"]
            assert node["source_row_content_sha256"]


def test_tampered_origin_and_residual_are_rejected(packet: dict, config: dict) -> None:
    bad_origin = copy.deepcopy(packet)
    bad_origin["dynamic_11_rows"][0]["equation_origin"]["source_equation_row"] = 9
    with pytest.raises(System10CommonTubeFullRHSError, match="seal/replay"):
        _verify_candidate_packet(bad_origin, 0, config)

    bad_residual = copy.deepcopy(packet)
    bad_residual["dynamic_11_rows"][0]["exact_residual_replay"]["zero"] = False
    with pytest.raises(System10CommonTubeFullRHSError, match="seal/replay"):
        _verify_candidate_packet(bad_residual, 0, config)


def test_missing_row_cannot_be_called_complete(packet: dict, config: dict) -> None:
    incomplete = copy.deepcopy(packet)
    incomplete["dynamic_11_rows"].pop()
    with pytest.raises(System10CommonTubeFullRHSError, match="seal/replay"):
        _verify_candidate_packet(incomplete, 0, config)


def test_linked_lower_order_W_tamper_is_rejected(packet: dict, config: dict) -> None:
    tampered = copy.deepcopy(packet)
    tampered["dynamic_11_rows"][0]["rhs_W_nodes"][0]["source_W_entry_sha256"] = "0" * 64
    with pytest.raises(System10CommonTubeFullRHSError, match="seal/replay"):
        _verify_candidate_packet(tampered, 0, config)


def test_sealed_artifacts_replay_and_aggregate_exactly() -> None:
    receipt = build_receipt(CONFIG, OUTPUT, root=ROOT)
    checked = json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert receipt == checked
    assert receipt["decision"] == DECISION
    assert receipt["counts"] == {
        "candidates": 12,
        "candidate_passes": 12,
        "candidate_blocks": 0,
        "predecessor_rows_per_candidate": 74,
        "new_dynamic_rows_per_candidate": 11,
        "new_dynamic_row_instances": 132,
        "new_exact_zero_residual_replays": 132,
        "equation_origin_seals": 1020,
        "total_rhs_rows_per_candidate": 85,
        "total_rhs_row_instances": 1020,
        "full_rhs_candidates_closed_on_common_tube": 12,
    }
    assert receipt["claims"]["all_twelve_common_tube_exact_85_state_rhs_closed"] is True
    assert receipt["claims"]["fixed_r_positive_domain_full_rhs_closed"] is False
    assert receipt["claims"]["constraint_propagation_closed"] is False
    assert receipt["claims"]["hyperbolicity_closed"] is False
