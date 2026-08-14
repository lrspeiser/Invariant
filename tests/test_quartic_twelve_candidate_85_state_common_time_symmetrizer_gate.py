from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_twelve_candidate_85_state_common_time_symmetrizer_gate import (
    Quartic85StateSymmetrizerGateError,
    _canonical_sha,
    _nonzero_witness_control,
    _sylvester_contract,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_twelve_candidate_85_state_common_time_symmetrizer_gate.json"
OUTPUT = (
    ROOT / "runs/math/quartic-twelve-candidate-85-state-common-time-symmetrizer-gate/receipt.json"
)


def test_exact_typed_block_all_twelve() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "TYPED_BLOCK_RESONANT_SYLVESTER_AND_SCHUR_DOMAIN_UNREGISTERED"
    results = receipt["candidate_results"]
    assert len(results) == 12
    assert {item["outcome"] for item in results} == {"BLOCK"}
    assert all(len(item["reason_codes"]) == 2 for item in results)


def test_nonzero_coupling_witness_is_exactly_diagonalizable() -> None:
    witness = _nonzero_witness_control()
    assert witness["time_block_determinant"] == "19683/4096"
    assert witness["rational_root_geometric_multiplicities"] == {
        "-1": 8,
        "1": 8,
        "-1/2": 4,
        "1/2": 4,
        "-1/3": 4,
        "1/3": 4,
    }
    assert witness["fluid_plus_minus_inverse_sqrt_3_combined_dimension"] == 2
    assert witness["eigenspace_dimension_sum"] == 34
    assert witness["diagonalizable"] is True


def test_sylvester_and_positivity_contract_is_precise() -> None:
    contract = _sylvester_contract()
    assert contract["block_decomposition"]["cross_unknown_X_shape"] == [30, 55]
    assert contract["block_decomposition"]["cross_unknown_entries"] == 1650
    assert contract["cross_symmetry_equation"] == "M^T X-X G=H_m C"
    assert contract["shared_characteristic_roots"] == ["-1", "+1"]
    assert contract["resonant_scalar_control"]["left_side"] == "0"
    assert contract["resonant_scalar_control"]["unit_forcing_residual"] == "-1"
    assert len(contract["minimal_missing_registrations"]) == 4


def test_counts_claims_and_content_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["counts"] == {
        "candidates": 12,
        "vacuum_K55_prerequisites_passed": 12,
        "matter_common_time_prerequisites_passed": 12,
        "nonzero_coupling_diagonalizable_witnesses": 1,
        "sylvester_unknown_entries_per_candidate": 1650,
        "resonant_roots_requiring_compatibility": 2,
        "coupled_symmetrizers_passed": 0,
        "typed_blocks": 12,
        "constraint_propagation_claims": 0,
        "rejects": 0,
    }
    claims = receipt["claims"]
    assert claims["nonzero_coupling_witness_diagonalizable"] is True
    assert not any(
        value for name, value in claims.items() if name != "nonzero_coupling_witness_diagonalizable"
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
    "binding",
    ["coupled_85_state_reduction", "vacuum_K55", "matter_common_time", "maxwell_mixed_block"],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateSymmetrizerGateError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["vacuum_K55"]["path"] = "runs/math/missing-k55.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateSymmetrizerGateError, match="cannot read bound file"):
        build_receipt(candidate, root=ROOT)


def test_broadened_symmetrizer_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["full_coupled_symmetrizer"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateSymmetrizerGateError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
