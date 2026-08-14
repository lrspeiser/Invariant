from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_sourced_gravity_constraint_propagation_gate import (
    Quartic85StateSourcedGravityConstraintError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_sourced_gravity_constraint_propagation_gate.json"
OUTPUT = ROOT / (
    "runs/math/quartic-85-state-sourced-gravity-constraint-propagation-gate/receipt.json"
)


def test_maximal_flat_reference_result_is_typed_block() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == (
        "TYPED_BLOCK_CANDIDATE_GRAVITY_CONSTRAINT_JET_DIVERGENCE_UNREGISTERED"
    )
    closed = receipt["materialization"]["closed_flat_reference_subgates"]
    assert closed["completed_85_state_symmetrizer_bound"] is True
    assert closed["total_matter_source_divergence_cancels_on_shell"] is True
    assert closed["candidates_with_identical_source_cancellation"] == 12
    assert closed["maxwell_subsidiary_equation"] == "box_g C=0"
    assert closed["vacuum_definition_time_residuals"] == ["0", "0", "0"]
    assert closed["vacuum_curl_time_residuals"] == ["0", "0", "0"]


def test_exact_source_cancellation_and_completeness_negative() -> None:
    replay = build_receipt(CONFIG, root=ROOT)["materialization"]["source_cancellation_replay"]
    assert replay["normalized_source_divergence_coefficients"] == [
        "-1/2",
        "-1/2",
        "-1/2",
    ]
    assert replay["on_shell_source_divergence_residual"] == "0"
    negative = replay["omitted_fluid_coefficient_negative"]
    assert negative["coefficient_residual"] == ["0", "0", "1/2"]
    assert negative["rejected"] is True
    assert "not a gravity subsidiary-system residual" in negative["scope"]


def test_missing_gravity_identity_is_not_inferred_from_symmetrizer() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    inference = receipt["materialization"]["gravity_constraint_inference"]
    assert inference["conclusion"] == "BLOCK"
    assert inference["reason_code"] == (
        "missing_candidate_gravity_constraint_jet_divergence_registration"
    )
    assert inference["premises"]["bounded_flat_85_state_symmetrizer"] is True
    assert (
        inference["premises"]["candidate_gauge_fixed_euler_divergence_identity_registered"] is False
    )
    registrations = receipt["materialization"]["minimal_registration_contract"]
    assert len(registrations) == 5
    assert {entry["registration"] for entry in registrations} == {
        "candidate_gravity_constraint_basis",
        "candidate_gauge_fixed_euler_divergence_identity",
        "flat_subsidiary_factorization",
        "constraint_surface_initial_data_map",
        "sourced_constraint_corruption_witness",
    }


def test_checked_receipt_and_claims_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    claims = receipt["claims"]
    assert claims["flat_reference_matter_source_divergence_cancellation_closed"] is True
    assert claims["sourced_gravity_constraint_propagation_closed"] is False
    assert claims["candidate_jet_uniformity_closed"] is False
    assert claims["nonlinear_global_closure_established"] is False
    assert claims["gravity_h7_theorem_established"] is False
    assert claims["universal_all_matter_closure_established"] is False
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
        "bounded_flat_symmetrizer",
        "matter_interface",
        "sourced_metric_euler",
        "vacuum_first_order_constraints",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateSourcedGravityConstraintError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["matter_interface"]["path"] = "runs/math/absent.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateSourcedGravityConstraintError, match="cannot read"):
        build_receipt(candidate, root=ROOT)


def test_broadened_gravity_constraint_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["sourced_gravity_constraint_propagation"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateSourcedGravityConstraintError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
