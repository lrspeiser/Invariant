from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_gauge_map_scalar_coefficient_expansion_gate import (
    Quartic85StateGaugeMapScalarExpansionError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_gauge_map_scalar_coefficient_expansion_gate.json"
OUTPUT = ROOT / (
    "runs/math/quartic-85-state-gauge-map-scalar-coefficient-expansion-gate/receipt.json"
)


def test_four_flat_scalar_rows_are_expanded_and_hash_bound() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == (
        "BOUNDED_PASS_FLAT_SCALAR_ROWS_TYPED_BLOCK_GENERAL_EXTERNAL_JETS"
    )
    rows = receipt["materialization"]["flat_constant_scalar_rows"]
    assert len(rows) == 4
    assert all(item["nonzero_scalar_coefficients"] > 0 for item in rows)
    assert all(len(item["entries"]) == item["nonzero_scalar_coefficients"] for item in rows)
    assert all(
        item["row_sha256"]
        == _canonical_sha({key: value for key, value in item.items() if key != "row_sha256"})
        for item in rows
    )
    candidates = receipt["materialization"]["candidate_results"]
    assert len(candidates) == 12
    assert len({item["manifest_sha256"] for item in candidates}) == 12
    assert all(item["expanded_rows"] == 4 for item in candidates)


def test_coefficients_are_exact_and_lowered_to_85_operators() -> None:
    rows = build_receipt(CONFIG, root=ROOT)["materialization"]["flat_constant_scalar_rows"]
    entries = [entry for row in rows for entry in row["entries"]]
    assert any("sqrt(2)" in entry["coefficient"] for entry in entries)
    assert all(17 <= entry["state_index"] <= 77 for entry in entries)
    assert all(len(entry["remaining_derivative_operator"]) == 2 for entry in entries)
    assert len({row["row_sha256"] for row in rows}) == 4


def test_three_negatives_reject_coefficient_corruption_and_zero_fill() -> None:
    negatives = build_receipt(CONFIG, root=ROOT)["materialization"]["negative_controls"]
    assert negatives["wrong_hat_time_coefficient"]["differing_coefficients"] > 0
    assert negatives["wrong_hat_time_coefficient"]["rejected"] is True
    assert negatives["wrong_off_diagonal_basis"]["differing_coefficients"] > 0
    assert negatives["wrong_off_diagonal_basis"]["rejected"] is True
    assert negatives["external_zero_fill"] == {
        "mutation": "set d2_H[0,0|0]=1 after omitting all external-jet columns",
        "exact_missing_divQ_lower_0_coefficient": "-9/4",
        "rejected": True,
    }


def test_general_value_packet_block_is_minimal_and_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    block = receipt["materialization"]["general_expansion_block"]
    assert block["reason_code"] == (
        "general_scalar_rows_require_external_and_lower_jet_value_packet"
    )
    assert [item["exact_scalar_values"] for item in block["required_packets"]] == [
        580,
        280,
        150,
        0,
    ]
    assert block["total_exact_scalar_values_before_domain"] == 1010
    assert block["zero_fill_forbidden"] is True
    claims = receipt["claims"]
    assert claims["exact_flat_reference_scalar_coefficient_rows_closed"] is True
    assert claims["general_external_jet_scalar_expansion_closed"] is False
    assert claims["constraint_propagation_closed"] is False
    assert claims["gravity_h7_theorem_established"] is False


def test_checked_receipt_is_path_free_and_content_addressed() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)
    checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert checked == receipt
    rendered = json.dumps(checked, sort_keys=True)
    assert str(ROOT) not in rendered
    assert "\\\\" not in rendered


@pytest.mark.parametrize(
    "binding",
    [
        "indexed_gauge_map",
        "vacuum_first_order_reference",
        "vacuum_nonlinear_euler",
        "constraint_coordinate_basis",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateGaugeMapScalarExpansionError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_general_expansion_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["general_external_jet_scalar_expansion"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateGaugeMapScalarExpansionError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
