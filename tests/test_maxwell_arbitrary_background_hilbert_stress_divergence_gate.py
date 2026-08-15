from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.maxwell_arbitrary_background_hilbert_stress_divergence_gate import (
    MaxwellArbitraryBackgroundStressError,
    _canonical_sha,
    _exact_arbitrary_local_jet_replay,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/maxwell_arbitrary_background_hilbert_stress_divergence_gate.json"


def test_closes_arbitrary_background_stress_divergence_only() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_ARBITRARY_BACKGROUND_MAXWELL_STRESS_DIVERGENCE"
    assert receipt["counts"] == {
        "dimensions": 4,
        "independent_field_strength_components": 6,
        "independent_potential_second_jets": 40,
        "antisymmetry_residuals": 16,
        "bianchi_residuals": 64,
        "stress_identity_components": 4,
        "stress_identity_residual_monomials": 0,
        "registered_controls": 4,
        "action_terms_specialized": 2,
        "negative_controls": 1,
        "negative_residual_components": 4,
        "blocks": 0,
        "rejects": 0,
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_exact_local_jet_identity_and_negative() -> None:
    replay = _exact_arbitrary_local_jet_replay()
    assert replay["field_strength_antisymmetry_all_zero"] is True
    assert replay["differential_bianchi_all_zero"] is True
    assert replay["identity_component_residual_monomials"] == [0, 0, 0, 0]
    assert replay["identity_components_all_zero"] is True
    assert replay["curvature_commutators_required"] is False
    negative = replay["negative_control"]
    assert negative["nonzero_components"] == 4
    assert negative["nonzero_monomials"] > 0
    assert negative["first_witness"] is not None
    assert negative["rejected"] is True


def test_claims_remain_bounded() -> None:
    claims = build_receipt(CONFIG, root=ROOT)["claims"]
    assert claims["arbitrary_background_maxwell_hilbert_stress_divergence_closed"] is True
    assert claims["registered_profile_controls_remain_corroboration_only"] is True
    for name, value in claims.items():
        if name not in {
            "arbitrary_background_maxwell_hilbert_stress_divergence_closed",
            "registered_profile_controls_remain_corroboration_only",
        }:
            assert value is False, name


@pytest.mark.parametrize(
    "binding",
    [
        "predecessor",
        "formal_controls",
        "proca_action_ir",
        "registered_covariant_identity_source",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if binding == "predecessor":
        config["predecessor"]["file_sha256"] = "0" * 64
    else:
        config["evidence_bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(MaxwellArbitraryBackgroundStressError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_policy_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["coupled_gravity_matter_pde"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(MaxwellArbitraryBackgroundStressError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
