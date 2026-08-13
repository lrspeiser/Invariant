from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.barotropic_irrotational_stress_conservation_gate import (
    BarotropicStressConservationError,
    _canonical_sha,
    _specialized_noether_replay,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/barotropic_irrotational_stress_conservation_gate.json"


def test_closes_second_gate_only() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_SECOND_GATE_ONLY"
    assert [item["outcome"] for item in receipt["gate_results"]] == [
        "PREDECESSOR_PASS",
        "PASS",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
    ]
    assert receipt["counts"] == {
        "sectors": 1,
        "predecessor_gates": 1,
        "new_gates_passed": 1,
        "gates_not_evaluated": 2,
        "registered_exact_residuals": 4,
        "specialized_exact_residual_coefficients": 3,
        "negative_controls": 2,
        "blocks": 0,
        "rejects": 0,
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_specialized_noether_replay_and_negative() -> None:
    replay = _specialized_noether_replay()
    assert replay["off_shell_identity_residual"] == [0, 0, 0]
    assert replay["divergence_coefficients"] == [-2, 2, 0]
    assert replay["euler_times_gradient_coefficients"] == [-2, 2, 0]
    assert replay["negative_control"]["residual"] != [0, 0, 0]
    assert replay["negative_control"]["rejected"] is True


def test_claims_remain_bounded() -> None:
    claims = build_receipt(CONFIG, root=ROOT)["claims"]
    assert claims["action_level_gate_closed_by_predecessor"] is True
    assert claims["stress_conservation_gate_closed"] is True
    for name, value in claims.items():
        if name not in {
            "action_level_gate_closed_by_predecessor",
            "stress_conservation_gate_closed",
        }:
            assert value is False, name


@pytest.mark.parametrize("binding", ["predecessor", "formal_controls", "variation_source"])
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if binding == "predecessor":
        config["predecessor"]["file_sha256"] = "0" * 64
    else:
        config["evidence_bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BarotropicStressConservationError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_policy_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["vortical_fluid"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BarotropicStressConservationError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
