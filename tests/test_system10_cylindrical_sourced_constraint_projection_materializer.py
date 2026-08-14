from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_sourced_constraint_projection_materializer import (
    System10CylindricalConstraintProjectionError,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "system10_cylindrical_sourced_constraint_projection_materializer.json"
RECEIPT = (
    ROOT
    / "runs"
    / "math"
    / "system10-cylindrical-sourced-constraint-projection-materializer"
    / "receipt.json"
)


def test_committed_receipt_replays_exactly() -> None:
    assert build_receipt(CONFIG) == json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_r1_projection_operator_has_exact_normalization_and_source_map() -> None:
    receipt = build_receipt(CONFIG)
    operator = receipt["materialization"]["projection_operator"]
    rows = operator["rows"]
    assert [row["constraint_row"] for row in rows] == [
        "Hamiltonian_E_nn",
        "momentum_E_n1",
        "momentum_E_n2",
        "momentum_E_n3",
    ]
    assert [row["sourced_metric_euler_row"] for row in rows] == [0, 1, 2, 3]
    assert [row["projection_coefficient"] for row in rows] == [
        "1",
        "sqrt(2)/2",
        "sqrt(2)/2",
        "sqrt(2)/2",
    ]
    assert [row["projected_source"] for row in rows] == [
        "-T_total^00/2",
        "-T_total^01/2",
        "-T_total^02/2",
        "-T_total^03/2",
    ]
    assert len({row["projection_sha256"] for row in rows}) == 4


def test_all_48_candidate_skeletons_are_bound_but_no_coordinate_row_is_inferred() -> None:
    receipt = build_receipt(CONFIG)
    resumable = receipt["materialization"]["candidate_resumable_packets"]
    packets = resumable["packets"]
    assert len(packets) == 48
    assert len({packet["candidate_id"] for packet in packets}) == 12
    assert len({packet["packet_sha256"] for packet in packets}) == 48
    assert all(packet["projection_skeleton_closed"] for packet in packets)
    assert not any(packet["coordinate_differential_row_closed"] for packet in packets)
    assert all(packet["first_missing_primitive"]["zero_inference_forbidden"] for packet in packets)
    assert receipt["counts"]["physical_gravity_coordinate_rows_closed"] == 48
    assert receipt["counts"]["hamiltonian_momentum_coordinate_rows_closed"] == 0
    assert receipt["claims"]["sourced_acceleration_cancellation_closed"] is False
    assert resumable["atomicity"].startswith("a candidate advances only")


def test_independent_geometric_controls_fix_all_four_projection_signs() -> None:
    controls = build_receipt(CONFIG)["materialization"]["independent_geometric_controls"]
    assert controls["passed"] == 5 and controls["failed"] == 0
    observed = {
        item["control_id"]: item["observed_exact_values"] for item in controls["controls"]
    }
    assert observed == {
        "flat_cylindrical_known_answer": ["0", "0", "0", "0"],
        "hamiltonian_second_radial_jet_mutation": ["-1", "0", "0", "0"],
        "radial_momentum_mixed_jet_mutation": ["0", "1/2", "0", "0"],
        "angular_momentum_mixed_jet_mutation": ["0", "0", "1/2", "0"],
        "axial_momentum_mixed_jet_mutation": ["0", "0", "0", "1/2"],
    }


def test_normalization_source_sign_and_partial_advance_negatives_reject() -> None:
    negatives = build_receipt(CONFIG)["materialization"]["negative_controls"]
    assert negatives["momentum_normalization"]["exact_difference"] == "(sqrt(2)-1)/2"
    assert negatives["momentum_normalization"]["rejected"] is True
    assert negatives["source_sign"]["independent_sector_residual_coefficients"] == [
        "1",
        "1",
        "1",
    ]
    assert negatives["partial_candidate_advance"]["observed_rows"] == 1
    assert all(item["rejected"] for item in negatives.values())


def test_claim_broadening_and_predecessor_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["hamiltonian_momentum_coordinate_rows"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalConstraintProjectionError, match="claims policy"):
        build_receipt(broadened, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["sourced_metric_euler"]["file_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalConstraintProjectionError, match="hash mismatch"):
        build_receipt(tampered, root=ROOT)


def test_missing_bound_file_fails_typed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["constraint_basis"]["path"] = "runs/math/missing.json"
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalConstraintProjectionError, match="cannot read"):
        build_receipt(missing, root=ROOT)
