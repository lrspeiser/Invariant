from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_open_r_propagation_decomposition_audit import (
    DECISION,
    _sealed,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_open_r_propagation_decomposition_audit.json"
CHECKED = (
    ROOT / "runs/math/system10-cylindrical-open-r-propagation-decomposition-audit/receipt.json"
)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt(CONFIG, root=ROOT)


def test_all_first_spatial_rhs_jets_are_closed(receipt: dict) -> None:
    assert receipt["counts"]["radial_rhs_jets_bound"] == 132
    assert receipt["counts"]["tangential_rhs_jets_bound"] == 264
    assert receipt["counts"]["all_first_spatial_rhs_jets_bound"] == 396
    assert receipt["claims"]["all_first_spatial_rhs_jets_closed"] is True
    assert (
        receipt["materialization"]["closed_predecessor_block"]["closed_candidate_instances"] == 264
    )


def test_reaudit_binds_rhs_origins_constraints_divq_and_identity(receipt: dict) -> None:
    assert receipt["counts"]["full_rhs_rows_bound"] == 1020
    assert receipt["counts"]["equation_origin_seals_bound"] == 1020
    assert receipt["counts"]["physical_gravity_rows_bound"] == 96
    assert receipt["counts"]["divQ_rows_bound"] == 4
    witness = receipt["materialization"]["coordinate_decomposition_witness"]
    assert witness["covariant_identity"].startswith("2*nabla^mu E_sourced_mu_nu")


def test_first_missing_primitive_is_exact_and_complete(receipt: dict) -> None:
    missing = receipt["materialization"]["first_missing_primitive"]
    assert missing["status"] == "BLOCK_48_COORDINATE_DECOMPOSITION_ROWS_UNREGISTERED"
    assert missing["required_row_count"] == 48
    assert len(missing["required_rows"]) == 48
    assert {(row["candidate_index"], row["lower_nu"]) for row in missing["required_rows"]} == {
        (candidate, nu) for candidate in range(12) for nu in range(4)
    }
    assert len(missing["required_terms"]) == 6


def test_audit_blocks_without_propagation_overclaim(receipt: dict) -> None:
    assert receipt["decision"] == DECISION
    assert receipt["counts"]["coordinate_decomposition_rows_bound"] == 0
    assert receipt["claims"]["coordinate_off_shell_decomposition_closed"] is False
    assert receipt["claims"]["constraint_propagation_closed"] is False
    assert receipt["claims"]["hyperbolicity_closed"] is False


def test_projection_origin_and_omission_controls_reject(receipt: dict) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "infer_decomposition_from_origin_hashes",
        "flip_normal_projection_sign",
        "omit_spatial_euler_divergence",
    }
    assert all(control["rejected"] is True for control in controls.values())


def test_checked_receipt_replays_and_tamper_breaks_seal(receipt: dict) -> None:
    checked = json.loads(CHECKED.read_text(encoding="utf-8"))
    assert receipt == checked
    tampered = copy.deepcopy(checked)
    tampered["claims"]["constraint_propagation_closed"] = True
    assert _sealed(tampered) is False
