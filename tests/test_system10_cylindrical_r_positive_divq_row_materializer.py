from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import sympy as sp

from sigma_theory_compiler.system10_cylindrical_r_positive_divq_row_materializer import (
    R,
    System10CylindricalDivQError,
    _physical_tensors,
    _portable_text_sha,
    build_receipt,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_divq_row_materializer.json"
RECEIPT = ROOT / "runs/math/system10-cylindrical-r-positive-divq-row-materializer/receipt.json"


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return build_receipt(CONFIG)


def _term(row: dict[str, Any], state: int, derivatives: list[int]) -> dict[str, Any]:
    return next(
        item
        for item in row["terms"]
        if item["state_index"] == state and item["coordinate_derivatives"] == derivatives
    )


def test_committed_four_row_receipt_replays_exactly(receipt: dict[str, Any]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["decision"] == (
        "BOUNDED_PASS_FOUR_R_POSITIVE_DIVQ_ROWS_BLOCK_FULL_EVOLUTION_RHS"
    )
    assert receipt["counts"]["divq_rows_registered"] == 4
    assert receipt["counts"]["constraint_propagation_proofs"] == 0


def test_all_four_rows_are_explicit_nonzero_85_state_operators(
    receipt: dict[str, Any],
) -> None:
    rows = receipt["materialization"]["rows"]
    assert [row["component"] for row in rows] == [f"divQ_lower[{index}]" for index in range(4)]
    assert [row["term_count"] for row in rows] == [43, 55, 50, 43]
    assert all(row["maximum_coordinate_derivative_order"] == 2 for row in rows)
    assert all(row["normalization"].endswith("/M2") for row in rows)
    assert all(0 <= term["state_index"] < 85 for row in rows for term in row["terms"])


def test_fixed_cylindrical_tensor_signs_and_row_coefficients_are_exact(
    receipt: dict[str, Any],
) -> None:
    metric, inverse, hat, connection = _physical_tensors()
    assert metric[2][2] == R**2
    assert inverse[2][2] == R**-2
    assert hat[0][0] == -9
    assert connection[1][2][2] == -R
    assert connection[2][1][2] == 1 / R

    rows = receipt["materialization"]["rows"]
    assert _term(rows[0], 1, [0, 0])["coefficient"] == "9/(4*r)"
    assert _term(rows[0], 1, [1, 1])["coefficient"] == "-1/(4*r)"
    assert _term(rows[1], 5, [2])["coefficient"] == "1/(2*r**4)"
    assert _term(rows[2], 5, [])["coefficient"] == "-3/(4*r**3)"


def test_denominator_pole_and_r1_replay_certificates_close(
    receipt: dict[str, Any],
) -> None:
    materialization = receipt["materialization"]
    pole = materialization["pole_certificate"]
    assert pole["domain"] == "r>0"
    assert pole["maximum_denominator_r_power"] == 5
    assert pole["denominator_zero_set"] == ["r=0"]
    assert pole["poles_on_admitted_domain"] == 0
    assert receipt["counts"]["r1_rows_replayed"] == 4
    for row in materialization["rows"]:
        assert row["r1_terms_sha256"]
        for symbolic, at_one in zip(row["terms"], row["r1_terms"], strict=True):
            value = sp.sympify(symbolic["coefficient"], locals={"r": R}).subs(R, 1)
            assert sp.factor(value - sp.sympify(at_one["coefficient"])) == 0


def test_zero_fill_sign_drop_and_axis_corruptions_fail_closed(
    receipt: dict[str, Any],
) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "zero_fill_all_four_rows",
        "drop_last_term",
        "flip_completion_sign",
        "include_axis",
    }
    assert all(control["rejected"] is True for control in controls.values())
    assert (
        controls["zero_fill_all_four_rows"]["mutated_row_set_sha256"]
        != receipt["materialization"]["row_set_sha256"]
    )


def test_candidate_manifests_stop_at_the_next_registered_gap(
    receipt: dict[str, Any],
) -> None:
    candidates = receipt["materialization"]["candidate_results"]
    assert len(candidates) == 12
    assert len({item["candidate_id"] for item in candidates}) == 12
    assert all(item["registered_divQ_rows"] == 4 for item in candidates)
    assert all(item["constraint_propagation_claimed"] is False for item in candidates)
    assert receipt["materialization"]["next_missing_primitive"]["status"] == (
        "BLOCK_PRINCIPAL_ONLY"
    )
    assert receipt["claims"]["full_nonlinear_85_state_rhs_closed"] is False
    assert receipt["claims"]["constraint_propagation_closed"] is False


def test_line_endings_are_normalized_but_non_line_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    source = Path(__file__).read_bytes()
    lf = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    expected = hashlib.sha256(lf).hexdigest()
    alternate = tmp_path / "alternate.py"
    alternate.write_bytes(crlf)
    assert _portable_text_sha(alternate) == expected
    alternate.write_bytes(crlf + b"# scientific tamper\r\n")
    assert _portable_text_sha(alternate) != expected


def test_binding_expectation_claim_and_output_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["r_positive_domain"]["file_sha256"] = "0" * 64
    tampered_binding = tmp_path / "tampered-binding.json"
    tampered_binding.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalDivQError, match="hash mismatch"):
        build_receipt(tampered_binding, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["frozen_expectations"]["term_counts"][0] += 1
    tampered_expectation = tmp_path / "tampered-expectation.json"
    tampered_expectation.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalDivQError, match="expectations"):
        build_receipt(tampered_expectation, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["constraint_propagation"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalDivQError, match="claims policy"):
        build_receipt(broadened, root=ROOT)

    conflict = tmp_path / "receipt.json"
    conflict.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10CylindricalDivQError, match="immutable output conflict"):
        write_receipt(CONFIG, conflict, root=ROOT)
