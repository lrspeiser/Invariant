from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.barotropic_irrotational_hyperbolicity_gate import (
    BarotropicHyperbolicityGateError,
    _canonical_sha,
    _exact_specialized_principal_replay,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/barotropic_irrotational_hyperbolicity_gate.json"


def test_closes_third_gate_only() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_THIRD_GATE_ONLY"
    assert [item["outcome"] for item in receipt["gate_results"]] == [
        "PREDECESSOR_PASS",
        "PREDECESSOR_PASS",
        "PASS",
        "NOT_EVALUATED",
    ]
    assert receipt["counts"] == {
        "sectors": 1,
        "predecessor_gates": 2,
        "new_gates_passed": 1,
        "gates_not_evaluated": 1,
        "exact_registered_residuals": 4,
        "exact_specialized_residuals": 2,
        "registered_negative_controls": 4,
        "specialized_negative_controls": 2,
        "blocks": 0,
        "rejects": 0,
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_exact_acoustic_cone_energy_and_negatives() -> None:
    replay = _exact_specialized_principal_replay()
    assert replay["aligned_integer_coefficients"] == [-6, 2, 2, 2]
    assert replay["determinant_coefficient"] == -48
    assert replay["determinant_residual"] == 0
    assert replay["sound_speed_squared"] == "1/3"
    assert replay["cone_residual"] == 0
    assert replay["positive_rescaled_hamiltonian_coefficients"] == [12, 1]
    assert all(item["rejected"] is True for item in replay["negative_controls"].values())


def test_claims_remain_bounded() -> None:
    claims = build_receipt(CONFIG, root=ROOT)["claims"]
    assert claims["action_and_stress_gates_closed_by_predecessors"] is True
    assert claims["irrotational_matter_hyperbolicity_gate_closed"] is True
    for name, value in claims.items():
        if name not in {
            "action_and_stress_gates_closed_by_predecessors",
            "irrotational_matter_hyperbolicity_gate_closed",
        }:
            assert value is False, name


@pytest.mark.parametrize("binding", ["predecessor", "formal_controls", "principal_source"])
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if binding == "predecessor":
        config["predecessor"]["file_sha256"] = "0" * 64
    else:
        config["evidence_bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BarotropicHyperbolicityGateError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_policy_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["coupled_gravity_matter_hyperbolicity"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BarotropicHyperbolicityGateError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
