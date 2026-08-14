from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_open_r_propagation_reaudit import (
    DECISION,
    _sealed,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_open_r_propagation_reaudit.json"
CHECKED = ROOT / "runs/math/system10-cylindrical-open-r-propagation-reaudit/receipt.json"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt(CONFIG, root=ROOT)


def test_radial_block_is_closed_for_all_twelve(receipt: dict) -> None:
    closed = receipt["materialization"]["closed_predecessor_block"]
    assert closed["status"] == "PASS_ALL_TWELVE_RADIAL_RHS_JETS"
    assert closed["closed_candidate_instances"] == 132
    assert receipt["claims"]["all_twelve_radial_rhs_jets_closed"] is True


def test_tangential_chain_rule_witness_is_exact(receipt: dict) -> None:
    witness = receipt["materialization"]["tangential_chain_rule_witness"]
    assert witness["constraint_row"] == "momentum_E_n1"
    assert witness["chain_rule"] == "partial_0(partial_2 v_5)=partial_2 F_5"
    assert witness["exact_unreplayed_partial_2_F_5_coefficient"] == "-sqrt(2)/256"
    assert witness["registered_partial_1_F_5"] is True
    assert witness["registered_partial_2_F_5"] is False
    assert witness["nonzero"] is True


def test_reaudit_blocks_without_overclaim(receipt: dict) -> None:
    assert receipt["decision"] == DECISION
    assert receipt["counts"]["radial_rhs_jets_bound"] == 132
    assert receipt["counts"]["tangential_rhs_jets_required"] == 264
    assert receipt["counts"]["tangential_rhs_jets_bound"] == 0
    assert receipt["counts"]["candidate_subsidiary_systems_closed"] == 0
    assert receipt["claims"]["constraint_propagation_closed"] is False
    assert receipt["claims"]["hyperbolicity_closed"] is False


def test_first_missing_primitive_is_complete_and_bounded(receipt: dict) -> None:
    missing = receipt["materialization"]["first_missing_primitive"]
    assert missing["status"] == "BLOCK_TANGENTIAL_TOTAL_DERIVATIVE_DAGS_UNREGISTERED"
    assert missing["required_directions"] == [2, 3]
    assert len(missing["required_outputs"]) == 22
    assert missing["required_candidate_instances"] == 264
    assert missing["required_W_derivative_nodes"] == ["D2_W_0..D2_W_10", "D3_W_0..D3_W_10"]


def test_zero_fill_and_direction_alias_controls_reject(receipt: dict) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert controls["reuse_partial_1_as_partial_2"]["rejected"] is True
    assert controls["zero_fill_tangential_jets"]["rejected"] is True
    assert controls["zero_fill_tangential_jets"]["nonzero_witness_coefficient"] == "-sqrt(2)/256"


def test_checked_receipt_replays_and_tamper_breaks_seal(receipt: dict) -> None:
    checked = json.loads(CHECKED.read_text(encoding="utf-8"))
    assert receipt == checked
    tampered = copy.deepcopy(checked)
    tampered["counts"]["tangential_rhs_jets_bound"] = 264
    assert _sealed(tampered) is False
