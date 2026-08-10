import json
from pathlib import Path

from sigma_theory_compiler.quartic_component_jacobian_contract_campaign import (
    generic_component_jacobian_contract_control,
    run_quartic_component_jacobian_contract_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
GOOD_UNKNOWN = (
    RUNS / "quartic-paradifferential-good-unknown-campaign" / "campaign.json"
)
SOLVED_SOURCE = RUNS / "quartic-solved-source-moser-campaign" / "campaign.json"
NONLINEAR = RUNS / "quartic-nonlinear-evolution-campaign" / "campaign.json"
PDE = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_component_jacobian_contract_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-component-jacobian-contract-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_sparse_injection_kinematics_and_negative_controls_are_exact() -> None:
    passed, control = generic_component_jacobian_contract_control()
    assert passed
    injection = control["principal_jet_injection"]
    assert injection["shape"] == [153, 55]
    assert injection["nonzero_entry_count"] == 132
    assert injection["duplicate_position_count"] == 0
    assert set(control["kinematic_evolution_rows"]["residuals"].values()) == {"0"}
    assert set(
        control["curl_compatible_spatial_Hessian"]["residuals"].values()
    ) == {"0"}
    assert control["curl_compatible_spatial_Hessian"]["negative_rejected"]
    assert control["solved_acceleration_mixed_derivative"]["negative_rejected"]
    assert control["norm_envelope_insufficiency_negative"]["rejected"]


def test_all_candidates_receive_exact_schema_but_dynamic_packet_is_missing() -> None:
    result = run_quartic_component_jacobian_contract_campaign(
        _load(GOOD_UNKNOWN),
        _load(SOLVED_SOURCE),
        _load(NONLINEAR),
        _load(PDE),
        _load(CONFIG),
    )
    assert result["status"] == (
        "pass_all_12_component_jacobian_schema_audits_packet_missing_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "jet_injections_certified": 12,
        "component_identities_proved": 0,
        "remainder_bounds_proved": 0,
        "global_H7_summations_applied": 0,
        "rejected": 0,
    }
    assert all(
        item["principal_jet_injection"]["nonzero_entries"] == 132
        and not item["component_packet_validation"]["present"]
        and not item["D_Y_E55_times_J_equals_iP55_proved"]
        and not item["paralinearization_remainder_bound_proved"]
        and not item["H7_derivative_loss_resolved"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_false_identity_and_corrupt_provenance_reject() -> None:
    campaigns = tuple(map(_load, (GOOD_UNKNOWN, SOLVED_SOURCE, NONLINEAR, PDE)))
    config = _load(CONFIG)
    false_identity = dict(config)
    false_identity["declare_component_identity_proved"] = True
    result = run_quartic_component_jacobian_contract_campaign(
        *campaigns, false_identity
    )
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[2]))
    corrupt["certificates"][0]["candidate_id"] = "corrupt"
    result = run_quartic_component_jacobian_contract_campaign(
        campaigns[0], campaigns[1], corrupt, campaigns[3], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
