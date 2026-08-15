from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_gauge_row_specialization_materializer import (
    System10CylindricalGaugeMaterializerError,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_gauge_row_specialization_materializer.json"
RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-gauge-row-specialization-materializer/receipt.json"
)


def test_committed_receipt_replays_exactly() -> None:
    assert build_receipt(CONFIG, root=ROOT) == json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_all_1010_cylindrical_values_are_exact_and_22_are_nonzero() -> None:
    packet = build_receipt(CONFIG, root=ROOT)["materialization"]["specialized_value_packet"]
    assert packet["scalar_values"] == 1010
    assert packet["nonzero_values"] == 22
    assert packet["zero_values"] == 988
    assert packet["all_values_exact"] is True
    assert packet["general_value_packet"] is False
    family_counts = {item["family"]: item["nonzero_values"] for item in packet["families"]}
    assert sum(family_counts[name] for name in list(family_counts)[:4]) == 3
    assert sum(family_counts[name] for name in list(family_counts)[4:11]) == 13
    assert sum(family_counts[name] for name in list(family_counts)[11:]) == 6


def test_four_gauge_rows_close_for_all_twelve_but_adm_stays_blocked() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    rows = receipt["materialization"]["shared_modified_harmonic_rows"]
    assert len(rows) == 4
    assert all(row["evaluated_row_value"] == "0" for row in rows)
    assert all(row["coefficient_support"] > 0 for row in rows)
    assert len(receipt["materialization"]["candidate_results"]) == 12
    assert receipt["counts"]["modified_harmonic_rows_closed_all_candidates"] == 48
    assert receipt["counts"]["physical_gravity_rows_closed"] == 48
    assert receipt["counts"]["physical_gravity_rows_required"] == 96
    assert receipt["counts"]["hamiltonian_momentum_rows_closed"] == 0
    assert receipt["claims"]["all_96_physical_gravity_rows_closed"] is False
    assert receipt["claims"]["general_coupled_hyperbolicity_closed"] is False
    assert receipt["claims"]["sourced_constraint_propagation_closed"] is False


def test_reference_jet_and_row_drop_negatives_reject() -> None:
    negatives = build_receipt(CONFIG, root=ROOT)["materialization"]["negative_controls"]
    assert negatives["omit_reference_connection"]["correct_value"] == "0"
    assert negatives["omit_reference_connection"]["corrupted_value"] == "-1"
    assert negatives["omit_reference_connection"]["rejected"] is True
    assert negatives["corrupt_nonzero_jet_value"]["exact_delta"] == "1"
    assert negatives["corrupt_nonzero_jet_value"]["rejected"] is True
    assert negatives["drop_candidate_gauge_row"]["observed_specialized_rows"] == 47
    assert negatives["drop_candidate_gauge_row"]["rejected"] is True


def test_broadened_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["general_common_domain"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalGaugeMaterializerError, match="claims policy"):
        build_receipt(path, root=ROOT)


def test_predecessor_tamper_and_missing_file_fail_typed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["curvilinear_source_control"]["file_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalGaugeMaterializerError, match="hash mismatch"):
        build_receipt(tampered, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["curvilinear_source_control"]["path"] = "runs/absent-cylindrical.json"
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalGaugeMaterializerError, match="cannot read bound file"):
        build_receipt(missing, root=ROOT)
