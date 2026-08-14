from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.system10_cylindrical_r_positive_constraint_propagation_attempt import (
    System10CylindricalPropagationError,
    build_receipt,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_constraint_propagation_attempt.json"
RECEIPT = (
    ROOT / "runs/math/system10-cylindrical-r-positive-constraint-propagation-attempt/receipt.json"
)


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return build_receipt(CONFIG)


def test_committed_typed_block_replays_exactly(receipt: dict[str, Any]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["decision"] == (
        "TYPED_BLOCK_R_POSITIVE_DIVQ_ROWS_AND_FULL_EVOLUTION_UNREGISTERED"
    )


def test_new_r_positive_authority_closes_all_external_divq_jet_values(
    receipt: dict[str, Any],
) -> None:
    closed = receipt["materialization"]["closed_inputs"]
    external = closed["external_jet_closure"]
    assert closed["physical_gravity_rows"] == 96
    assert external["domain"] == "r>0"
    assert external["external_formulation_jet_slots_required"] == 580
    assert external["external_formulation_jet_slots_valued"] == 580
    assert all(item["closed"] is True for item in external["families"].values())


def test_first_missing_primitive_is_four_expanded_divq_rows(
    receipt: dict[str, Any],
) -> None:
    missing = receipt["materialization"]["first_missing_primitive"]
    assert missing["primitive"] == (
        "sourced_cylindrical_r_positive_divQ_lower_four_component_85_state_"
        "rational_differential_operator_rows"
    )
    assert missing["required_rows"] == 4
    assert missing["registered_rows"] == 0
    assert missing["inputs_already_closed"]["r_positive_external_jet_slots"] == 580
    assert missing["inputs_already_closed"]["physical_metric_third_85_operator_slots"] == 200
    assert missing["acceptance"]["zero_fill_forbidden"] is True


def test_principal_only_rhs_is_exactly_nonidentifiable(receipt: dict[str, Any]) -> None:
    witness = receipt["materialization"]["lower_order_nonidentifiability_witness"]
    next_missing = receipt["materialization"]["next_missing_primitive"]
    assert witness["principal_A_B_C_change"] == "0"
    assert witness["nonzero"] is True
    assert witness["constraint_time_derivative_delta"] != "0"
    assert witness["state_coordinate"].startswith("v_")
    assert next_missing["principal_rows_registered"] == 85
    assert next_missing["full_lower_order_rhs_rows_registered"] == 0
    assert next_missing["exact_nonidentifiability_witness_sha256"] == witness["witness_sha256"]


def test_domain_energy_and_corruption_controls_remain_fail_closed(
    receipt: dict[str, Any],
) -> None:
    energy = receipt["materialization"]["energy_control"]
    controls = receipt["materialization"]["negative_controls"]
    assert energy["domain_denominators_certified"] is True
    assert energy["domain"] == "r>0"
    assert energy["subsidiary_operator_available"] is False
    assert energy["energy_functional_constructed"] is False
    assert set(controls) == {
        "infer_divQ_zero",
        "zero_fill_physical_third_operator",
        "treat_principal_matrix_as_full_rhs",
        "include_axis",
        "borrow_parent_energy",
    }
    assert all(item["rejected"] is True for item in controls.values())


def test_no_propagation_or_broader_theorem_is_claimed(receipt: dict[str, Any]) -> None:
    claims = receipt["claims"]
    assert claims["r_positive_rows_and_domain_closed"] is True
    assert claims["candidate_bound_subsidiary_system_closed"] is False
    assert claims["sourced_constraint_propagation_closed"] is False
    assert claims["subsidiary_energy_estimate_closed"] is False
    assert claims["arbitrary_formulation_functions_closed"] is False
    assert claims["general_hyperbolicity_closed"] is False
    assert claims["global_theorem_established"] is False
    assert claims["promotion_authorized"] is False


def test_tamper_claim_broadening_and_immutable_output_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["indexed_divergence_map"]["file_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalPropagationError, match="hash mismatch"):
        build_receipt(tampered, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["sourced_constraint_propagation"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalPropagationError, match="claims policy"):
        build_receipt(broadened, root=ROOT)

    conflict = tmp_path / "receipt.json"
    conflict.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10CylindricalPropagationError, match="immutable output conflict"):
        write_receipt(CONFIG, conflict, root=ROOT)
