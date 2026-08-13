from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.barotropic_irrotational_action_gate import (
    BarotropicActionGateError,
    _canonical_sha,
    _fluid_replay,
    build_receipt,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/barotropic_irrotational_action_gate.json"


def test_closes_earliest_action_gate_only() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_EARLIEST_GATE_ONLY"
    assert [item["outcome"] for item in receipt["gate_results"]] == [
        "PASS",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
    ]
    assert receipt["admitted_action"]["dependencies"] == ["g_mu_nu", "kappa", "tau"]
    assert receipt["admitted_action"]["forbidden_gravitational_or_species_dependencies"] == []
    assert receipt["counts"] == {
        "sectors": 1,
        "gates_passed": 1,
        "gates_not_evaluated": 3,
        "exact_integer_residuals": 2,
        "exact_registered_variation_residuals": 4,
        "negative_controls": 1,
        "registered_formal_controls": 1,
        "blocks": 0,
        "rejects": 0,
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_exact_barotropic_and_stress_replay_has_negative() -> None:
    replay = _fluid_replay()
    assert replay["equation_of_state"] == "p=rho/3"
    assert replay["equation_of_state_residual"] == "0"
    assert replay["stress_decomposition_residual"] == "0"
    assert replay["negative_control"]["equation_of_state_residual"] != "0"
    assert replay["negative_control"]["rejected"] is True


def test_claims_are_narrow() -> None:
    claims = build_receipt(CONFIG, root=ROOT)["claims"]
    assert claims["earliest_action_level_gate_closed"] is True
    for name, value in claims.items():
        if name != "earliest_action_level_gate_closed":
            assert value is False, name


def test_checked_receipt_replays_and_resealed_later_gate_pass_is_rejected() -> None:
    checked = json.loads(
        (ROOT / "runs/math/barotropic-irrotational-action-gate/receipt.json").read_text(
            encoding="utf-8"
        )
    )
    validate_receipt(checked, CONFIG, root=ROOT)
    checked["gate_results"][1]["outcome"] = "PASS"
    checked["gate_results"][1]["reason_codes"] = []
    body = {key: value for key, value in checked.items() if key != "content_sha256"}
    checked["content_sha256"] = _canonical_sha(body)
    with pytest.raises(BarotropicActionGateError, match="immutable replay"):
        validate_receipt(checked, CONFIG, root=ROOT)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("universal_matter", "file_sha256"),
        ("maxwell_followup", "file_sha256"),
    ],
)
def test_predecessor_hash_tamper_fails_closed(tmp_path: Path, section: str, key: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["predecessors"][section][key] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BarotropicActionGateError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_claim_policy_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["universal_matter"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BarotropicActionGateError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
