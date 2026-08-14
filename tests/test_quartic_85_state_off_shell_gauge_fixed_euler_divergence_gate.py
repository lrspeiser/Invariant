from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_off_shell_gauge_fixed_euler_divergence_gate import (
    Quartic85StateOffShellEulerDivergenceError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_off_shell_gauge_fixed_euler_divergence_gate.json"
OUTPUT = ROOT / (
    "runs/math/quartic-85-state-off-shell-gauge-fixed-euler-divergence-gate/receipt.json"
)


def test_common_off_shell_formula_is_closed_for_all_twelve() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == (
        "BOUNDED_PASS_COMMON_COVARIANT_IDENTITY_TYPED_BLOCK_DIFFERENTIATED_GAUGE_MAP"
    )
    materialization = receipt["materialization"]
    formula = materialization["common_formula"]
    assert formula["maximal_common_off_shell_identity"] == (
        "2*nabla^mu E_sourced_mu_nu+E_phi_g*nabla_nu(phi_g)+F_total_nu-2*nabla^mu Q_mu_nu=0"
    )
    assert materialization["common_formula_sha256"] == _canonical_sha(formula)
    candidates = materialization["candidate_results"]
    assert len(candidates) == 12
    assert len({item["manifest_sha256"] for item in candidates}) == 12
    assert all(item["common_covariant_identity_closed"] is True for item in candidates)
    assert all(
        item["differentiated_gauge_completion_85_state_map_closed"] is False for item in candidates
    )


def test_source_sign_normalization_replay_and_negative() -> None:
    replay = build_receipt(CONFIG, root=ROOT)["materialization"]["normalization_replay"]
    assert replay["assembled_identity_residual"] == "0"
    negative = replay["wrong_source_sign_negative"]
    assert negative == {
        "mutation": "replace -T_total/2 by +T_total/2",
        "residual": "14",
        "expected_symbolic_residual": "2*F_total_nu",
        "rejected": True,
    }


def test_differentiated_gauge_source_map_is_precisely_blocked() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    block = receipt["materialization"]["differentiated_gauge_source_block"]
    assert block["reason_code"] == (
        "missing_differentiated_modified_harmonic_formulation_field_map"
    )
    assert block["available_gauge_jet_order"] == {
        "hat_inverse_metric": 0,
        "tilde_inverse_metric": 1,
        "reference_connection": 1,
        "gauge_source_H_beta": 1,
        "physical_metric": 2,
    }
    assert len(block["required_for_nabla_Q"]) == 6
    assert block["zero_fill_forbidden"] is True
    claims = receipt["claims"]
    assert claims["common_off_shell_covariant_sourced_identity_closed"] is True
    assert claims["differentiated_gauge_completion_in_85_state_variables_closed"] is False
    assert claims["constraint_propagation_closed"] is False
    assert claims["candidate_jet_uniformity_closed"] is False
    assert claims["gravity_h7_theorem_established"] is False


def test_checked_receipt_is_path_free_and_content_addressed() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
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
        "constraint_coordinate_basis",
        "sourced_metric_euler",
        "vacuum_gauge_fixed_euler",
        "off_shell_noether_controls",
        "matter_divergence_interface",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateOffShellEulerDivergenceError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["vacuum_gauge_fixed_euler"]["path"] = "runs/math/absent.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateOffShellEulerDivergenceError, match="cannot read"):
        build_receipt(candidate, root=ROOT)


def test_broadened_differentiated_map_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["differentiated_gauge_completion_in_85_state_variables"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateOffShellEulerDivergenceError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
