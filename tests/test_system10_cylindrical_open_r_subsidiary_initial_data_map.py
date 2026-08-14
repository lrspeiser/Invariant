from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_open_r_subsidiary_initial_data_map import (
    DECISION,
    _sealed,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_open_r_subsidiary_initial_data_map.json"
OUTPUT = ROOT / "runs/math/system10-cylindrical-open-r-subsidiary-initial-data-map"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt(CONFIG, root=ROOT)


def test_all_twelve_initial_data_maps_close(receipt: dict) -> None:
    assert receipt["decision"] == DECISION
    assert receipt["counts"]["candidate_subsidiary_initial_data_maps_closed"] == 12
    assert len(receipt["candidate_results"]) == 12
    assert all(
        item["outcome"].startswith("PASS_INITIAL_DATA_MAP") for item in receipt["candidate_results"]
    )


def test_gravity_normal_map_is_exact_and_invertible(receipt: dict) -> None:
    gravity = receipt["materialization"]["gravity_normal_derivative_map"]
    assert gravity["A"] == [
        ["-9/4", "0", "0", "0"],
        ["0", "9/4", "0", "0"],
        ["0", "0", "9/(4*r**2)", "0"],
        ["0", "0", "0", "9/4"],
    ]
    assert gravity["det_A"] == "-6561/(256*r**2)"
    assert gravity["det_A_nonzero_on_r_positive"] is True
    assert gravity["vanishing_H_M_implies_four_vanishing_normal_C_derivatives"] is True


def test_maxwell_gauss_and_lorenz_map_replays_reduced_normal_row(receipt: dict) -> None:
    maxwell = receipt["materialization"]["maxwell_normal_derivative_map"]
    assert len(maxwell["lorenz_state_operator_terms"]) == 5
    assert len(maxwell["gauss_state_operator_terms"]) == 8
    assert maxwell["symbolic_identity_residual"] == "0"
    assert maxwell["normal_derivative_map"] == ("partial_0(C_Maxwell)=-Maxwell_Gauss when E_L_0=0")


def test_homogeneous_system_is_sealed_but_propagation_stays_blocked(receipt: dict) -> None:
    system = receipt["materialization"]["homogeneous_subsidiary_system"]
    assert system["closed_equation_count"] == 5
    assert system["cauchy_uniqueness_proved"] is False
    blocker = receipt["materialization"]["propagation_audit"]["first_missing_primitive"]
    assert blocker["status"] == "BLOCK_SUBSIDIARY_ENERGY_OR_UNIQUENESS_CERTIFICATE_UNREGISTERED"
    assert blocker["registered_energy_estimates"] == 0
    assert receipt["claims"]["constraint_propagation_closed"] is False


def test_invalid_initial_data_and_propagation_shortcuts_reject(receipt: dict) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "pointwise_C_instead_of_slice_field",
        "drop_Maxwell_Gauss",
        "corrupt_Q_normal_sign",
        "claim_propagation_without_uniqueness",
    }
    assert all(item["rejected"] is True for item in controls.values())


def test_checked_receipt_replays_and_tamper_fails(receipt: dict) -> None:
    assert receipt == json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert _sealed(receipt)
    tampered = copy.deepcopy(receipt)
    tampered["counts"]["candidate_subsidiary_initial_data_maps_closed"] = 11
    assert not _sealed(tampered)
