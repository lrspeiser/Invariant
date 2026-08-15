from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_open_r_twelve_candidate_rhs_jets import (
    DECISION,
    System10OpenRTwelveCandidateRHSJetsError,
    _verify_candidate,
    build_candidate_packet,
    build_receipt,
)
from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _load_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_open_r_twelve_candidate_rhs_jets.json"
OUTPUT = ROOT / "runs/math/system10-cylindrical-open-r-twelve-candidate-rhs-jets"


@pytest.fixture(scope="module")
def packet_11() -> dict:
    return build_candidate_packet(CONFIG, 11, root=ROOT)


def test_late_candidate_closes_exact_open_r_rows_and_jets(packet_11: dict) -> None:
    assert packet_11["candidate_index"] == 11
    assert len(packet_11["rows"]) == 11
    assert packet_11["replay"] == {
        "open_r_zero_residuals": 11,
        "radially_differentiated_zero_residuals": 11,
        "exact_r1_rhs_replays": 11,
    }
    assert packet_11["open_r_neighborhood"]["determinant_r_scaling"] == "r**(-10)"
    assert packet_11["open_r_neighborhood"]["denominators_nonzero"] is True
    assert packet_11["claims"]["candidate_open_r_all_11_radial_rhs_jets_closed"] is True
    assert packet_11["claims"]["constraint_propagation_closed"] is False


def test_all_checked_packets_are_atomic_and_sealed() -> None:
    config = _load_json(CONFIG)
    for index in range(12):
        packet = json.loads((OUTPUT / f"candidate-{index:02d}.json").read_text(encoding="utf-8"))
        _verify_candidate(packet, config, index)
        assert packet["candidate_index"] == index
        assert len(packet["W_radial_derivative_nodes"]) == 11
        assert all(
            node["unclassified_atom_count"] == 0 for node in packet["W_radial_derivative_nodes"]
        )


def test_aggregate_receipt_replays_all_twelve() -> None:
    receipt = build_receipt(CONFIG, OUTPUT, root=ROOT)
    checked = json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert receipt == checked
    assert receipt["decision"] == DECISION
    assert receipt["counts"]["candidate_passes"] == 12
    assert receipt["counts"]["candidate_blocks"] == 0
    assert receipt["counts"]["open_r_rhs_rows"] == 132
    assert receipt["counts"]["open_r_radial_rhs_jets"] == 132
    assert receipt["counts"]["radially_differentiated_zero_residuals"] == 132
    assert receipt["counts"]["remaining_candidate_packets"] == 0
    assert receipt["claims"]["all_twelve_open_r_radial_rhs_jet_primitive_closed"] is True
    assert receipt["claims"]["constraint_propagation_closed"] is False


def test_candidate_margins_are_positive_and_bound() -> None:
    receipt = json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    margins = [
        item["exact_absolute_determinant_lower_bound"] for item in receipt["candidate_results"]
    ]
    assert len(margins) == 12
    assert all(value != "0" for value in margins)
    assert all(item["packet_content_sha256"] for item in receipt["candidate_results"])


def test_candidate_and_jet_tamper_fail_closed(packet_11: dict) -> None:
    config = _load_json(CONFIG)
    wrong_candidate = copy.deepcopy(packet_11)
    wrong_candidate["candidate_index"] = 10
    with pytest.raises(System10OpenRTwelveCandidateRHSJetsError, match="packet seal"):
        _verify_candidate(wrong_candidate, config, 11)
    missing = copy.deepcopy(packet_11)
    missing["rows"].pop()
    with pytest.raises(System10OpenRTwelveCandidateRHSJetsError, match="packet seal"):
        _verify_candidate(missing, config, 11)


def test_scope_does_not_promote_propagation() -> None:
    receipt = json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["counts"]["constraint_propagation_proofs"] == 0
    assert receipt["claims"]["constraint_propagation_closed"] is False
    assert receipt["claims"]["hyperbolicity_closed"] is False
    assert receipt["claims"]["promotion_authorized"] is False
