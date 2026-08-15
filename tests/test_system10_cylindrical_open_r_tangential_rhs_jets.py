from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_open_r_tangential_rhs_jets import (
    DECISION,
    System10OpenRTangentialRHSJetsError,
    _directional_derivative_atom,
    _verify_packet,
    build_candidate_packet,
    build_receipt,
)
from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _load_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_open_r_tangential_rhs_jets.json"
OUTPUT = ROOT / "runs/math/system10-cylindrical-open-r-tangential-rhs-jets"


@pytest.fixture(scope="module")
def packet_11() -> dict:
    return build_candidate_packet(CONFIG, 11, root=ROOT)


def test_late_candidate_closes_22_tangential_jets(packet_11: dict) -> None:
    assert packet_11["candidate_index"] == 11
    assert len(packet_11["W_directional_derivative_nodes"]) == 22
    assert len(packet_11["tangential_rhs_jets"]) == 22
    assert packet_11["replay"] == {
        "base_rhs_rows_replayed": 11,
        "tangential_rhs_jets": 22,
        "differentiated_zero_residuals": 22,
        "directions": [2, 3],
    }
    assert packet_11["claims"]["candidate_all_22_tangential_rhs_jets_closed"] is True
    assert packet_11["claims"]["constraint_propagation_closed"] is False


def test_directional_atom_contract_is_exact() -> None:
    assert _directional_derivative_atom("q_4", 2) == "w_2_4"
    assert _directional_derivative_atom("v_7", 3) == "partial_3_v_7"
    assert _directional_derivative_atom("w_1_3", 2) == "partial_2_w_1_3"
    assert _directional_derivative_atom("partial_3_v_2", 2) == "partial_2_partial_3_v_2"
    assert _directional_derivative_atom("partial_2_w_3_8", 3) == "partial_3_partial_2_w_3_8"
    assert _directional_derivative_atom("unknown", 2) is None


def test_all_checked_packets_are_atomic_and_classified() -> None:
    config = _load_json(CONFIG)
    for index in range(12):
        packet = json.loads((OUTPUT / f"candidate-{index:02d}.json").read_text(encoding="utf-8"))
        _verify_packet(packet, config, index)
        assert all(
            node["unclassified_atom_count"] == 0
            for node in packet["W_directional_derivative_nodes"]
        )


def test_aggregate_receipt_replays_264_jets() -> None:
    receipt = build_receipt(CONFIG, OUTPUT, root=ROOT)
    checked = json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert receipt == checked
    assert receipt["decision"] == DECISION
    assert receipt["counts"]["candidate_passes"] == 12
    assert receipt["counts"]["direction_2_rhs_jets"] == 132
    assert receipt["counts"]["direction_3_rhs_jets"] == 132
    assert receipt["counts"]["tangential_rhs_jets"] == 264
    assert receipt["counts"]["differentiated_zero_residuals"] == 264
    assert receipt["counts"]["unclassified_W_atoms"] == 0
    assert (
        receipt["claims"]["all_twelve_first_spatial_rhs_jets_closed_with_radial_predecessor"]
        is True
    )
    assert receipt["claims"]["constraint_propagation_closed"] is False


def test_tamper_and_missing_direction_fail_closed(packet_11: dict) -> None:
    config = _load_json(CONFIG)
    tampered = copy.deepcopy(packet_11)
    tampered["tangential_rhs_jets"][0]["expression"] = "0"
    with pytest.raises(System10OpenRTangentialRHSJetsError, match="packet seal"):
        _verify_packet(tampered, config, 11)
    missing = copy.deepcopy(packet_11)
    missing["tangential_rhs_jets"] = [
        jet for jet in missing["tangential_rhs_jets"] if jet["direction"] == 2
    ]
    with pytest.raises(System10OpenRTangentialRHSJetsError, match="packet seal"):
        _verify_packet(missing, config, 11)


def test_scope_does_not_promote_propagation() -> None:
    receipt = json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["counts"]["constraint_propagation_proofs"] == 0
    assert receipt["claims"]["constraint_propagation_closed"] is False
    assert receipt["claims"]["hyperbolicity_closed"] is False
    assert receipt["claims"]["promotion_authorized"] is False
