from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_common_tube_propagation_audit import (
    DECISION,
    _sealed,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_common_tube_propagation_audit.json"
CHECKED = ROOT / "runs/math/system10-cylindrical-common-tube-propagation-audit/receipt.json"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt(CONFIG, root=ROOT)


def test_exact_radial_jet_nonidentifiability_witness(receipt: dict) -> None:
    witness = receipt["materialization"]["radial_jet_nonidentifiability_witness"]
    assert witness["constraint_row"] == "momentum_E_n1"
    assert witness["dynamic_row"] == "evolution_v[7]"
    assert witness["same_registered_tube_value"] is True
    assert witness["radial_derivative_delta_at_r_1"] == "1"
    assert witness["exact_constraint_time_derivative_delta"] == "1/128"
    assert witness["nonzero"] is True


def test_audit_binds_all_closed_inputs_but_blocks_propagation(receipt: dict) -> None:
    assert receipt["decision"] == DECISION
    assert receipt["counts"]["physical_gravity_rows_bound"] == 96
    assert receipt["counts"]["divQ_rows_bound"] == 4
    assert receipt["counts"]["full_rhs_candidate_packets_bound"] == 12
    assert receipt["counts"]["candidate_subsidiary_systems_closed"] == 0
    assert receipt["claims"]["full_85_state_rhs_closed_on_common_tube"] is True
    assert receipt["claims"]["radial_rhs_first_jet_identifiable"] is False
    assert receipt["claims"]["constraint_propagation_closed_on_common_tube"] is False
    assert receipt["claims"]["hyperbolicity_closed"] is False


def test_first_missing_primitive_is_sharp_and_bounded(receipt: dict) -> None:
    missing = receipt["materialization"]["first_missing_primitive"]
    assert missing["status"] == "BLOCK_RADIAL_RHS_JET_UNREGISTERED"
    assert missing["required_outputs"] == [f"partial_1 F_{index}" for index in range(11)]
    assert "open radial neighborhood" in missing["required_domain"]
    assert (
        missing["witness_sha256"]
        == receipt["materialization"]["radial_jet_nonidentifiability_witness"]["witness_sha256"]
    )


def test_checked_receipt_replays_deterministically(receipt: dict) -> None:
    assert receipt == json.loads(CHECKED.read_text(encoding="utf-8"))


def test_witness_tamper_breaks_receipt_seal() -> None:
    checked = json.loads(CHECKED.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(checked)
    tampered["materialization"]["radial_jet_nonidentifiability_witness"][
        "exact_constraint_time_derivative_delta"
    ] = "0"
    assert _sealed(tampered) is False
    assert tampered != build_receipt(CONFIG, root=ROOT)


def test_negative_controls_reject_point_derivative_inference(receipt: dict) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert controls["differentiate_point_value_as_constant"]["rejected"] is True
    assert controls["differentiate_point_value_as_constant"]["exact_missed_delta"] == "1/128"
    assert controls["claim_propagation_from_85_rows"]["rejected"] is True
