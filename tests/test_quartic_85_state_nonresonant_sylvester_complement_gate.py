from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_nonresonant_sylvester_complement_gate import (
    Quartic85StateNonresonantSylvesterError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_nonresonant_sylvester_complement_gate.json"
OUTPUT = ROOT / "runs/math/quartic-85-state-nonresonant-sylvester-complement-gate/receipt.json"


def test_exact_nonresonant_sylvester_solution_passes() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_EXACT_FLAT_SPHERE_NONRESONANT_SYLVESTER_COMPLEMENT"
    material = receipt["materialization"]
    assert material["sylvester_residual_sphere_normal_form_nonzero_entries"] == [
        0,
        0,
        0,
        0,
    ]
    assert len(material["solutions"]) == 4
    assert all(item["nonzero_polynomial_entries"] > 0 for item in material["solutions"])


def test_gap_census_and_exact_bounds_are_registered() -> None:
    material = build_receipt(CONFIG, root=ROOT)["materialization"]
    contract = material["spectral_contract"]
    assert contract["smallest_active_nonresonant_gap"] == "1/2"
    assert contract["largest_active_inverse_gap"] == "2"
    bounds = [
        Fraction(item["uniform_unit_sphere_infinity_norm_upper"]) for item in material["solutions"]
    ]
    assert all(bound > 0 for bound in bounds)
    assert Fraction(material["combined_arbitrary_B_infinity_norm_upper"]) == sum(bounds)


def test_zero_fluid_and_resonant_sectors_are_fail_closed() -> None:
    material = build_receipt(CONFIG, root=ROOT)["materialization"]
    assert material["spectral_contract"]["inactive_forcing_sectors"] == [
        "matter zero-speed projector",
        "fluid +/-1/sqrt(3) projectors",
    ]
    assert material["spectral_contract"]["forcing_row_support_in_30_state_basis"] == [
        19,
        20,
        21,
        22,
    ]
    assert all(
        record["gravity_root"] != record["matter_root"] for record in material["spectral_blocks"]
    )


def test_inverse_gap_corruption_is_rejected() -> None:
    negative = build_receipt(CONFIG, root=ROOT)["materialization"]["corruption_negative"]
    assert negative["rejected"] is True
    assert "inverse gap" in negative["mutation"]
    assert negative["residual_nonzero_polynomial_entries"] > 0
    assert negative["residual_normal_form_terms"] > 0


def test_claims_content_and_checked_receipt_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    claims = receipt["claims"]
    assert claims["exact_flat_sphere_nonresonant_sylvester_solution_closed"] is True
    assert claims["bounded_B_schur_positivity_closed"] is False
    assert claims["full_coupled_symmetrizer_closed"] is False
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
        "full_sphere_resonant_predecessor",
        "P55_sphere_pencil",
        "projector_recipes",
        "symmetrizer_blocker",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateNonresonantSylvesterError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["full_coupled_symmetrizer"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateNonresonantSylvesterError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
