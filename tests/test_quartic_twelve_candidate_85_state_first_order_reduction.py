from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_twelve_candidate_85_state_first_order_reduction import (
    Quartic85StateReductionError,
    _canonical_sha,
    _corruption_negative,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_twelve_candidate_85_state_first_order_reduction.json"
OUTPUT = ROOT / "runs/math/quartic-twelve-candidate-85-state-first-order-reduction/receipt.json"


def test_exact_85_state_reduction_passes_all_twelve() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE"
    certificate = receipt["reduction_certificate"]
    assert certificate["state"] == {"q_A": 17, "v_A": 17, "w_iA": 51, "total": 85}
    assert certificate["mass_shape"] == [85, 85]
    assert certificate["evolution_shape"] == [85, 85]
    assert certificate["nonzero_characteristic_lift_residual_zero"] is True
    results = receipt["candidate_results"]
    assert len(results) == 12
    assert {item["outcome"] for item in results} == {"PASS"}
    assert len({item["first_order_manifest_sha256"] for item in results}) == 12
    for item in results:
        assert item["first_order_manifest_sha256"] == _canonical_sha(item["first_order_manifest"])


def test_corrupted_kinematic_row_rejects_exactly() -> None:
    negative = _corruption_negative()
    assert negative["correct_residual"] == "0"
    assert negative["corrupted_lift_residual"] == "-30"
    assert negative["rejected"] is True


def test_counts_claims_and_content_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["counts"] == {
        "candidates": 12,
        "second_order_fields": 17,
        "first_order_states_per_candidate": 85,
        "first_order_state_entries_total": 1020,
        "directional_companion_dimension": 34,
        "zero_speed_auxiliary_modes_per_candidate": 51,
        "reductions_passed": 12,
        "lift_residual_entries": 0,
        "negative_controls": 1,
        "symmetrizers_constructed": 0,
        "constraint_propagation_claims": 0,
        "rejects": 0,
    }
    claims = receipt["claims"]
    assert claims["all_twelve_exact_85_state_first_order_reductions_closed"] is True
    assert not any(
        value
        for name, value in claims.items()
        if name != "all_twelve_exact_85_state_first_order_reductions_closed"
    )
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_checked_receipt_is_current_and_path_free() -> None:
    checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert checked == build_receipt(CONFIG, root=ROOT)
    rendered = json.dumps(checked, sort_keys=True)
    assert str(ROOT) not in rendered
    assert "\\\\" not in rendered


@pytest.mark.parametrize(
    "binding", ["complete_17_field_principal", "vacuum_reduction", "total_action"]
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateReductionError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["vacuum_reduction"]["path"] = "runs/math/missing-reduction.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateReductionError, match="cannot read bound file"):
        build_receipt(candidate, root=ROOT)


def test_broadened_symmetrizer_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["full_coupled_symmetrizer"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateReductionError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
