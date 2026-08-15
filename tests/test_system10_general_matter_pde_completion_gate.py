from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_general_matter_pde_completion_gate import (
    System10GeneralMatterPDECompletionError,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_general_matter_pde_completion_gate.json"
RECEIPT = ROOT / "runs/math/system10-general-matter-pde-completion-gate/receipt.json"


def test_committed_receipt_replays_exactly() -> None:
    assert build_receipt(CONFIG, root=ROOT) == json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_all_1010_general_value_slots_are_manifested_but_not_invented() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    attempt = receipt["materialization"]["general_scalar_value_attempt"]
    assert attempt["requested_scalar_values"] == 1010
    assert attempt["certified_general_values"] == 0
    assert attempt["missing_general_values"] == 1010
    assert attempt["zero_fill_forbidden"] is True
    assert [item["requested_scalar_values"] for item in attempt["packets"]] == [580, 280, 150]
    assert sum(item["requested_scalar_values"] for item in attempt["slot_families"]) == 1010
    assert len({slot for family in attempt["slot_families"] for slot in family["slot_ids"]}) == 1010


def test_96_physical_rows_and_general_pde_claims_fail_closed() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    rows = receipt["materialization"]["physical_gravity_row_attempt"]
    assert rows["rows_required"] == 96
    assert rows["modified_harmonic_rows_required"] == 48
    assert rows["hamiltonian_momentum_rows_required"] == 48
    assert rows["rows_closed"] == 0
    assert len(rows["required_rows"]) == 96
    claims = receipt["claims"]
    assert claims["matter_common_time_subgate_closed"] is True
    assert claims["flat_reference_bounded_symmetrizer_subgate_closed"] is True
    assert claims["general_common_domain_closed"] is False
    assert claims["all_96_physical_gravity_rows_closed"] is False
    assert claims["general_coupled_hyperbolicity_closed"] is False
    assert claims["general_common_time_positivity_closed"] is False
    assert claims["sourced_constraint_propagation_closed"] is False
    assert claims["gravity_h7_theorem_established"] is False
    assert claims["universal_all_matter_closure_established"] is False


def test_sign_domain_and_constraint_row_corruptions_are_rejected() -> None:
    negatives = build_receipt(CONFIG, root=ROOT)["materialization"]["negative_controls"]
    assert negatives["source_sign_corruption"]["sector_coefficient_deltas"] == ["1", "1", "1"]
    assert negatives["source_sign_corruption"]["rejected"] is True
    assert negatives["domain_omission"]["missing_requirements"] == [
        "external_jet_uniform_bounds_and_compatibility"
    ]
    assert negatives["domain_omission"]["rejected"] is True
    assert negatives["constraint_row_drop"]["observed_rows"] == 95
    assert negatives["constraint_row_drop"]["expected_rows"] == 96
    assert negatives["constraint_row_drop"]["rejected"] is True


def test_broadened_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["general_common_domain_closed"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GeneralMatterPDECompletionError, match="claims policy"):
        build_receipt(path, root=ROOT)


def test_predecessor_hash_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["indexed_gauge_map"]["file_sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GeneralMatterPDECompletionError, match="hash mismatch"):
        build_receipt(path, root=ROOT)


def test_missing_predecessor_fails_with_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["indexed_gauge_map"]["path"] = "runs/math/absent-system10.json"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GeneralMatterPDECompletionError, match="cannot read bound file"):
        build_receipt(path, root=ROOT)
