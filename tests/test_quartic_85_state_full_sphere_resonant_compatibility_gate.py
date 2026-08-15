from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_full_sphere_resonant_compatibility_gate import (
    Quartic85StateFullSphereCompatibilityError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_full_sphere_resonant_compatibility_gate.json"
OUTPUT = ROOT / "runs/math/quartic-85-state-full-sphere-resonant-compatibility-gate/receipt.json"


def test_full_sphere_resonant_reductions_are_exactly_zero() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_EXACT_FULL_SPHERE_RESONANT_COMPATIBILITY_FLAT_REFERENCE"
    reductions = receipt["materialization"]["resonant_sphere_reductions"]
    assert len(reductions) == 8
    assert all(item["sphere_normal_form_nonzero_entries"] == 0 for item in reductions)
    assert all(item["sphere_normal_form_terms"] == 0 for item in reductions)


def test_full_support_regression_distinguishes_restricted_control() -> None:
    regression = build_receipt(CONFIG, root=ROOT)["materialization"]["full_support_regression"]
    assert [item["outside_restricted_companion_support"] for item in regression] == [
        0,
        0,
        18,
        18,
        18,
        18,
    ]
    assert [item["difference_from_restricted_packet_nonzero_entries"] for item in regression] == [
        0,
        0,
        44,
        44,
        44,
        44,
    ]


def test_supersession_is_precise_and_non_mutating() -> None:
    supersession = build_receipt(CONFIG, root=ROOT)["materialization"]["supersession"]
    assert supersession["predecessor_e1_zero_conclusion_remains_valid"] is True
    assert supersession["predecessor_modified"] is False
    assert (
        "companion-restricted" in supersession["predecessor_serialized_projector_and_cross_packets"]
    )
    assert "full 55-state" in supersession["authoritative_representation_from_this_receipt"]


def test_full_cross_and_projector_manifests_are_materialized() -> None:
    material = build_receipt(CONFIG, root=ROOT)["materialization"]
    assert material["cross_axis_nonzero_counts"] == [
        [16, 16, 16],
        [20, 11, 11],
        [11, 20, 11],
        [11, 11, 20],
    ]
    assert set(material["full_vacuum_projectors"]) == {"-1", "1"}
    assert all(
        manifest["normal_form_terms"] == 1918
        for manifest in material["full_vacuum_projectors"].values()
    )
    assert material["full_projector_identity_reductions"] == [
        {
            "root": "-1",
            "idempotence_sphere_normal_form_nonzero_entries": 0,
            "eigenidentity_sphere_normal_form_nonzero_entries": 0,
        },
        {
            "root": "1",
            "idempotence_sphere_normal_form_nonzero_entries": 0,
            "eigenidentity_sphere_normal_form_nonzero_entries": 0,
        },
    ]
    assert len(material["full_maxwell_cross_pencils"]) == 4


def test_claims_content_and_checked_receipt_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    claims = receipt["claims"]
    assert claims["exact_unit_sphere_resonant_compatibility_closed"] is True
    assert claims["full_coupled_symmetrizer_closed"] is False
    assert claims["bounded_B_schur_positivity_closed"] is False
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
        "restricted_e1_predecessor",
        "P55_sphere_pencil",
        "projector_recipes",
        "maxwell_mixed_principal",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateFullSphereCompatibilityError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["full_coupled_symmetrizer"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateFullSphereCompatibilityError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
