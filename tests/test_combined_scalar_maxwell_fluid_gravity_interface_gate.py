from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.combined_scalar_maxwell_fluid_gravity_interface_gate import (
    CombinedMatterGravityInterfaceError,
    _canonical_sha,
    _exact_combined_replay,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/combined_scalar_maxwell_fluid_gravity_interface_gate.json"


def test_matter_interface_passes_and_gravity_interface_blocks() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == ("BOUNDED_PASS_MATTER_INTERFACE_WITH_TYPED_GRAVITY_BLOCK")
    assert [item["outcome"] for item in receipt["gate_results"]] == [
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "BLOCK",
    ]
    assert receipt["counts"] == {
        "matter_sectors": 3,
        "matter_second_order_components": 6,
        "light_cone_components": 5,
        "acoustic_cone_components": 1,
        "internal_matter_constraints": 1,
        "combined_interface_passes": 4,
        "gravity_interface_blocks": 1,
        "exact_combined_residuals": 4,
        "negative_controls": 2,
        "rejects": 0,
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_exact_combined_stress_principal_and_constraint_replay() -> None:
    replay = _exact_combined_replay()
    stress = replay["combined_stress_conservation"]
    assert stress["sector_euler_force_coefficients"] == [1, 1, 1]
    assert stress["on_shell_total_residual"] == [0, 0, 0]
    assert stress["negative_control"]["coefficient_residual"] == [0, 0, 1]
    principal = replay["combined_matter_principal_compatibility"]
    assert principal["second_order_components"] == 6
    assert principal["principal_block_coefficients"] == [
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-3, 1],
    ]
    assert principal["time_kinetic_coefficients"] == [1, 1, 1, 1, 1, 3]
    assert principal["strongly_hyperbolic_matter_direct_sum"] is True
    constraints = replay["internal_matter_constraint_closure"]
    assert constraints["ricci_field_contraction_residual_terms"] == 0
    assert constraints["corrupted_ricci_symmetry_negative"]["residual_terms"] == 1


def test_claims_and_minimal_gravity_registration_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    claims = receipt["claims"]
    assert claims["combined_three_sector_matter_interface_closed"] is True
    assert claims["total_matter_stress_conserved_on_shell"] is True
    assert claims["common_time_matter_principal_compatibility_closed"] is True
    assert claims["internal_matter_constraint_source_closure_closed"] is True
    for name, value in claims.items():
        if name not in {
            "combined_three_sector_matter_interface_closed",
            "total_matter_stress_conserved_on_shell",
            "common_time_matter_principal_compatibility_closed",
            "internal_matter_constraint_source_closure_closed",
        }:
            assert value is False, name
    block = receipt["gravity_block"]
    assert block["outcome"] == "BLOCK"
    assert len(block["minimal_registration_contract"]) == 6


@pytest.mark.parametrize(
    "binding",
    [
        "universal_matter",
        "maxwell_arbitrary_background",
        "fluid_constraint_complete",
        "covariant_field_contract",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if binding == "covariant_field_contract":
        config["evidence_bindings"][binding]["file_sha256"] = "0" * 64
    else:
        config["predecessors"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CombinedMatterGravityInterfaceError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_gravity_policy_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["pass_full_coupled_principal_system"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CombinedMatterGravityInterfaceError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
