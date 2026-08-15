from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_bounded_B_schur_symmetrizer_gate import (
    Quartic85StateBoundedBSymmetrizerError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_bounded_B_schur_symmetrizer_gate.json"
OUTPUT = ROOT / "runs/math/quartic-85-state-bounded-B-schur-symmetrizer-gate/receipt.json"


def test_corrected_full_symmetrizer_passes() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_EXACT_FLAT_SPHERE_FULL_SYMMETRIZER_BOUNDED_B"
    corrected = receipt["materialization"]["corrected_symmetrizer"]
    assert corrected["schur_complement"] == "H_g"
    assert corrected["gravity_energy_uniform_lower"] == "1/7"
    assert corrected["full_85_state_symmetry_residual_nonzero_entries"] == 0
    assert len(corrected["quadratic_correction_coefficients"]) == 10
    assert all(
        item["top_left_symmetry_residual_nonzero_entries"] == 0
        for item in corrected["quadratic_correction_coefficients"]
    )


def test_bounded_nonzero_domain_has_quantitative_lower_bound() -> None:
    domain = build_receipt(CONFIG, root=ROOT)["materialization"]["bounded_potential_domain"]
    assert domain == {
        "condition": "max_mu |B_mu| <= 8/38505",
        "contains_nonzero_potentials": True,
        "X_infinity_norm_upper": "1/15",
        "X_operator_norm_squared_upper": "2/15",
        "triangular_factor_norm_upper_strict": "2",
        "full_symmetrizer_uniform_lower_strict": "1/28",
        "schur_complement_uniform_lower": "1/7",
    }


def test_quadratic_correction_is_required() -> None:
    negative = build_receipt(CONFIG, root=ROOT)["materialization"]["corruption_negative"]
    assert negative["rejected"] is True
    assert "omit" in negative["mutation"]
    assert negative["top_left_residual_nonzero_polynomial_entries"] > 0
    assert negative["top_left_residual_normal_form_terms"] > 0


def test_claims_content_and_checked_receipt_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    claims = receipt["claims"]
    assert claims["exact_flat_sphere_full_85_state_symmetrizer_closed"] is True
    assert claims["candidate_jet_uniformity_closed"] is False
    assert claims["sourced_constraint_propagation_closed"] is False
    assert claims["gravity_h7_theorem_established"] is False
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)
    checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert checked == receipt
    rendered = json.dumps(checked, sort_keys=True)
    assert str(ROOT) not in rendered
    assert "\\\\" not in rendered


@pytest.mark.parametrize(
    "binding",
    ["nonresonant_solution", "P55_sphere_pencil", "projector_recipes", "symmetrizer_blocker"],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateBoundedBSymmetrizerError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_constraint_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["sourced_constraint_propagation"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateBoundedBSymmetrizerError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
