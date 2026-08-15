from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_open_r_representative_rhs_jets import (
    DECISION,
    System10OpenRRepresentativeRHSJetsError,
    _radial_derivative_atom,
    _verify_packet,
    build_packet,
    build_receipt,
)
from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _load_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_open_r_representative_rhs_jets.json"
OUTPUT = ROOT / "runs/math/system10-cylindrical-open-r-representative-rhs-jets"


@pytest.fixture(scope="module")
def packet() -> dict:
    return build_packet(CONFIG, root=ROOT)


def test_open_r_packet_closes_all_representative_rhs_jets(packet: dict) -> None:
    assert len(packet["rows"]) == 11
    assert len(packet["W_radial_derivative_nodes"]) == 11
    assert packet["replay"] == {
        "open_r_zero_residuals": 11,
        "radially_differentiated_zero_residuals": 11,
        "exact_r1_rhs_replays": 11,
    }
    assert packet["claims"]["representative_open_r_all_11_radial_rhs_jets_closed"] is True
    assert packet["claims"]["all_twelve_candidates_closed"] is False
    assert packet["claims"]["constraint_propagation_closed"] is False


def test_determinant_domain_and_r1_replay_are_exact(packet: dict) -> None:
    domain = packet["open_r_neighborhood"]
    assert domain["radial_interval"] == ["1/2", "3/2"]
    assert domain["real_v_10_interval"] == ["-1/4", "1/4"]
    assert domain["exact_absolute_determinant_lower_bound"] == "13/36"
    assert domain["denominators_nonzero"] is True
    assert "/(16384*r**10)" in domain["determinant"]
    assert all(row["r1_replay"] for row in packet["rows"])


def test_radial_derivative_atom_contract_is_closed() -> None:
    assert _radial_derivative_atom("q_4") == "w_1_4"
    assert _radial_derivative_atom("v_7") == "partial_1_v_7"
    assert _radial_derivative_atom("w_2_3") == "partial_1_w_2_3"
    assert _radial_derivative_atom("partial_3_v_2") == "partial_1_partial_3_v_2"
    assert _radial_derivative_atom("partial_2_w_3_8") == "partial_1_partial_2_w_3_8"
    assert _radial_derivative_atom("unknown_atom") is None


def test_W_nodes_have_no_unclassified_atoms(packet: dict) -> None:
    assert all(node["unclassified_atom_count"] == 0 for node in packet["W_radial_derivative_nodes"])
    assert all(node["classified_atom_count"] > 0 for node in packet["W_radial_derivative_nodes"])
    assert all(node["operator_sha256"] for node in packet["W_radial_derivative_nodes"])


def test_packet_tamper_and_missing_jet_fail_closed(packet: dict) -> None:
    config = _load_json(CONFIG)
    bad = copy.deepcopy(packet)
    bad["rows"][7]["radial_rhs_expression"] = "0"
    with pytest.raises(System10OpenRRepresentativeRHSJetsError, match="packet seal"):
        _verify_packet(bad, config)
    missing = copy.deepcopy(packet)
    missing["rows"].pop()
    with pytest.raises(System10OpenRRepresentativeRHSJetsError, match="packet seal"):
        _verify_packet(missing, config)


def test_checked_packet_and_receipt_replay() -> None:
    checked_packet = json.loads((OUTPUT / "candidate-00.json").read_text(encoding="utf-8"))
    checked_receipt = json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert checked_packet == build_packet(CONFIG, root=ROOT)
    assert checked_receipt == build_receipt(CONFIG, OUTPUT / "candidate-00.json", root=ROOT)
    assert checked_receipt["decision"] == DECISION
    assert checked_receipt["counts"]["open_r_radial_rhs_jets"] == 11
    assert checked_receipt["counts"]["remaining_candidate_packets"] == 11
    assert checked_receipt["scaling_plan"]["candidate_order"] == list(range(1, 12))
    assert checked_receipt["claims"]["constraint_propagation_closed"] is False
