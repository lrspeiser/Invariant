from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_maxwell_metric_mixed_principal_completion_gate import (
    MaxwellMetricMixedPrincipalCompletionError,
    _canonical_sha,
    _mixed_principal_replay,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_maxwell_metric_mixed_principal_completion_gate.json"
OUTPUT = ROOT / "runs/math/quartic-maxwell-metric-mixed-principal-completion-gate/receipt.json"


def test_exact_nonzero_mixed_block_and_all_twelve_principals_pass() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == ("PASS_EXACT_NONZERO_MAXWELL_MIXED_BLOCK_AND_17_FIELD_PRINCIPAL")
    replay = receipt["maxwell_metric_mixed_principal"]
    assert replay["matrix_shape"] == [4, 10]
    assert replay["structurally_nonzero_entries"] == 40
    assert replay["rnc_derivation"]["independent_expansion_residual_entries"] == 0
    assert replay["zero_block_negative"]["nonzero_residual_entries"] == 4
    results = receipt["candidate_results"]
    assert len(results) == 12
    assert {item["outcome"] for item in results} == {"PASS"}
    assert len({item["complete_17_field_principal_sha256"] for item in results}) == 12


def test_rnc_formula_and_sqrt_two_coordinate_replay() -> None:
    replay = _mixed_principal_replay()
    assert replay["rnc_derivation"]["action_euler_second_metric_derivative_block"] == ("0_(4x10)")
    assert replay["rnc_derivation"]["mixed_term"] == "-B_sigma partial_nu Gamma^sigma"
    matrix = replay["matrix_entries"]
    assert len(matrix) == 4 and all(len(row) == 10 for row in matrix)
    assert any("sqrt(2)" in entry for row in matrix for entry in row)
    witness = replay["zero_block_negative"]["witness_matrix"]
    assert sum(entry != "0" for row in witness for entry in row) == 4


def test_updated_census_claims_and_content_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["counts"] == {
        "candidates": 12,
        "mixed_block_rows": 4,
        "mixed_block_columns": 10,
        "mixed_block_entries": 40,
        "structurally_nonzero_mixed_entries": 40,
        "completed_17_field_principal_matrices": 12,
        "determined_entries_total": 3468,
        "unresolved_entries_total": 0,
        "rnc_replay_residual_entries": 0,
        "negative_controls": 1,
        "first_order_85_state_reductions": 0,
        "rejects": 0,
    }
    claims = receipt["claims"]
    assert claims["exact_maxwell_metric_mixed_principal_block_closed"] is True
    assert claims["all_twelve_17_field_second_order_principal_matrices_closed"] is True
    assert claims["mixed_block_universally_zero"] is False
    for name, value in claims.items():
        if name not in {
            "exact_maxwell_metric_mixed_principal_block_closed",
            "all_twelve_17_field_second_order_principal_matrices_closed",
        }:
            assert value is False, name
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_checked_receipt_is_current_and_path_free() -> None:
    checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert checked == build_receipt(CONFIG, root=ROOT)
    rendered = json.dumps(checked, sort_keys=True)
    assert str(ROOT) not in rendered
    assert "\\\\" not in rendered


@pytest.mark.parametrize("binding", ["blocked_principal_census", "total_action"])
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(MaxwellMetricMixedPrincipalCompletionError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["total_action"]["path"] = "runs/math/missing-total-action.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(MaxwellMetricMixedPrincipalCompletionError, match="cannot read bound file"):
        build_receipt(candidate, root=ROOT)


def test_broadened_first_order_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["full_85_state_first_order_reduction"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(MaxwellMetricMixedPrincipalCompletionError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
