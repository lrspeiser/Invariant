from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.maxwell_hilbert_noether_interface_gate import (
    MaxwellHilbertNoetherGateError,
    _canonical_sha,
    _zero_residuals,
    build_receipt,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/maxwell_hilbert_noether_interface_gate.json"


def test_gate_closes_only_the_dedicated_maxwell_interface() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "BOUNDED_PASS_WITH_TYPED_BLOCK"
    assert receipt["gate_id"] == "dedicated_maxwell_hilbert_stress_noether_identity"
    assert receipt["counts"] == {
        "registered_controls": 4,
        "action_terms_specialized": 2,
        "exact_noether_residuals": 12,
        "exact_structural_residuals": 1,
        "negative_controls": 1,
        "blocks": 1,
        "rejects": 0,
    }
    interface = receipt["noether_interface"]
    assert interface["minkowski_arbitrary_profile_residuals"] == ["0"] * 4
    assert interface["curved_profile_residuals"]["flrw_homogeneous"] == ["0"] * 4
    assert interface["curved_profile_residuals"]["static_spherical_radial"] == ["0"] * 4
    gauge = interface["gauge_divergence_identity"]
    assert gauge["contraction_residual"] == "0"
    assert gauge["corrupted_symmetry_negative_residual"] != "0"
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_claims_remain_bounded() -> None:
    claims = build_receipt(CONFIG, root=ROOT)["claims"]
    assert claims["dedicated_maxwell_registered_profile_interface_closed"] is True
    assert claims["dedicated_maxwell_arbitrary_background_interface_closed"] is False
    assert claims["curved_executable_profiles_are_universal_proof"] is False
    assert claims["universal_matter_closure_established"] is False
    assert claims["gravity_h7_theorem_established"] is False
    assert claims["global_boundary_control_established"] is False
    assert claims["promotion_authorized"] is False
    assert build_receipt(CONFIG, root=ROOT)["arbitrary_background_block"]["outcome"] == "BLOCK"


def test_checked_receipt_replays_and_resealed_overclaim_is_rejected() -> None:
    checked = json.loads(
        (ROOT / "runs/math/maxwell-hilbert-noether-interface-gate/receipt.json").read_text(
            encoding="utf-8"
        )
    )
    validate_receipt(checked, CONFIG, root=ROOT)
    checked["claims"]["dedicated_maxwell_arbitrary_background_interface_closed"] = True
    body = {key: value for key, value in checked.items() if key != "content_sha256"}
    checked["content_sha256"] = _canonical_sha(body)
    with pytest.raises(MaxwellHilbertNoetherGateError, match="immutable replay"):
        validate_receipt(checked, CONFIG, root=ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_predecessor",
        "bad_predecessor_hash",
        "bad_formal_hash",
        "bad_identity_hash",
        "bad_action_hash",
    ],
)
def test_missing_or_tampered_evidence_fails_closed(tmp_path: Path, mutation: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if mutation == "missing_predecessor":
        config["predecessor"]["path"] = "runs/math/absent.json"
    elif mutation == "bad_predecessor_hash":
        config["predecessor"]["file_sha256"] = "0" * 64
    elif mutation == "bad_formal_hash":
        config["evidence_bindings"]["formal_controls"]["file_sha256"] = "0" * 64
    elif mutation == "bad_identity_hash":
        config["evidence_bindings"]["covariant_identity_source"]["file_sha256"] = "0" * 64
    else:
        config["evidence_bindings"]["proca_action_ir"]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(MaxwellHilbertNoetherGateError):
        build_receipt(candidate, root=ROOT)


def test_registered_residual_tamper_fails_closed() -> None:
    with pytest.raises(MaxwellHilbertNoetherGateError, match="Minkowski residuals"):
        _zero_residuals(["1", "0", "0", "0"], 4, "Minkowski")
