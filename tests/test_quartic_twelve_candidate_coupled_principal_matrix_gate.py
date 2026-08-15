from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_twelve_candidate_coupled_principal_matrix_gate import (
    QuarticCoupledPrincipalMatrixGateError,
    _canonical_sha,
    _matrix_census,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_twelve_candidate_coupled_principal_matrix_gate.json"
OUTPUT = ROOT / "runs/math/quartic-twelve-candidate-coupled-principal-matrix-gate/receipt.json"


def test_exact_partial_skeleton_and_typed_block_all_twelve() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "TYPED_BLOCK_MISSING_MAXWELL_METRIC_MIXED_PRINCIPAL_BLOCK"
    results = receipt["candidate_results"]
    assert len(results) == 12
    assert {item["outcome"] for item in results} == {"BLOCK"}
    assert {item["reason_code"] for item in results} == {
        "missing_nonlinear_lorenz_maxwell_metric_mixed_principal_block"
    }
    for item in results:
        assert item["partial_matrix_skeleton_sha256"] == _canonical_sha(
            item["partial_matrix_skeleton"]
        )


def test_matrix_entry_census_is_exact() -> None:
    census = _matrix_census()
    assert census["block_sizes"] == {"gravity": 11, "scalar": 1, "maxwell": 4, "fluid": 1}
    assert census["second_order_dimension"] == 17
    assert census["target_first_order_state_dimension"] == 85
    assert census["diagonal_entries_determined"] == 139
    assert census["off_diagonal_zero_entries_determined"] == 110
    assert census["entries_determined"] == 249
    assert census["entries_unresolved"] == 40
    assert census["unresolved_block"]["shape"] == [4, 10]


def test_counts_claims_and_minimal_contract_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["counts"] == {
        "candidates": 12,
        "second_order_dimension": 17,
        "target_first_order_dimension": 85,
        "determined_entries_per_candidate": 249,
        "unresolved_entries_per_candidate": 40,
        "determined_entries_total": 2988,
        "unresolved_entries_total": 480,
        "full_matrices_passed": 0,
        "typed_blocks": 12,
        "rejects": 0,
    }
    claims = receipt["claims"]
    assert claims["exact_partial_matrix_skeleton_all_twelve"] is True
    assert not any(
        value
        for name, value in claims.items()
        if name != "exact_partial_matrix_skeleton_all_twelve"
    )
    contract = receipt["minimal_registration_contract"]
    assert contract["required_shape"] == [4, 10]
    assert "d E_Maxwell" in contract["required_derivative"]
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_checked_receipt_is_current_and_path_free() -> None:
    checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert checked == build_receipt(CONFIG, root=ROOT)
    rendered = json.dumps(checked, sort_keys=True)
    assert str(ROOT) not in rendered
    assert "\\\\" not in rendered


@pytest.mark.parametrize(
    "binding",
    [
        "sourced_metric_euler",
        "vacuum_first_order",
        "combined_matter",
        "universal_matter",
        "total_action",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticCoupledPrincipalMatrixGateError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["combined_matter"]["path"] = "runs/math/missing-matter-principal.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticCoupledPrincipalMatrixGateError, match="cannot read bound file"):
        build_receipt(candidate, root=ROOT)


def test_broadened_full_matrix_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["full_coupled_principal_matrix"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticCoupledPrincipalMatrixGateError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
