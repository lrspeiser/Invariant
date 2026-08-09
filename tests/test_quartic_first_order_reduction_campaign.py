from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.quartic_first_order_reduction_campaign import (
    generic_constraint_propagation_control,
    generic_scalar_first_order_reduction_control,
    quartic_full_first_order_reduction_control,
    run_quartic_first_order_reduction_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
SYMMETRIZER_PATH = RUNS / "quartic-symmetrizer-uniform-domain-campaign" / "campaign.json"
MOSER_PATH = RUNS / "quartic-quasilinear-moser-campaign" / "campaign.json"
CONFIG_PATH = ROOT / "configs" / "backgrounds" / "quartic_first_order_reduction_campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_generic_scalar_first_order_determinant_and_constraints_are_exact() -> None:
    scalar_passed, scalar = generic_scalar_first_order_reduction_control()
    constraint_passed, constraints = generic_constraint_propagation_control()
    assert scalar_passed
    assert scalar["determinant_residual"] == "0"
    assert scalar["negative_control"]["rejected"]
    assert constraint_passed
    assert set(constraints["definition_time_residuals"]) == {"0"}
    assert set(constraints["curl_time_residuals_in_coordinate_chart"]) == {"0"}
    assert constraints["negative_control"]["rejected"]


def test_quartic_full_first_order_reduction_reconstructs_companion() -> None:
    passed, evidence = quartic_full_first_order_reduction_control()
    assert passed
    assert evidence["first_order_state"]["total"] == 55
    assert evidence["directional_companion"]["dimension"] == 22
    assert evidence["spatial_block_extraction"]["B_reconstruction_residual_zero"]
    assert evidence["spatial_block_extraction"]["C_reconstruction_residual_zero"]
    assert evidence["full_pencil"]["nonzero_characteristic_lift_residual_zero"]
    assert evidence["full_pencil"]["directional_companion_lift_residual_zero"]


def test_all_quartic_candidates_bind_to_55_variable_reduction() -> None:
    result = run_quartic_first_order_reduction_campaign(
        _load(SYMMETRIZER_PATH), _load(MOSER_PATH), _load(CONFIG_PATH)
    )
    assert result["status"] == "pass_all_12_exact_55_variable_principal_first_order_reductions"
    assert result["counts"] == {
        "selected": 12,
        "exact_55_variable_reductions_passed": 12,
        "rejected": 0,
    }
    assert result["negative_controls"]["omitted_spatial_derivative_evolution"]["rejected"]
    assert result["negative_controls"]["omitted_definition_constraint_evolution"]["rejected"]
    assert all(
        item["state_dimensions"]["physical_space_first_order"] == 55
        and item["state_dimensions"]["directional_companion"] == 22
        and item["constraint_counts"] == {
            "derivative_definition": 33,
            "independent_spatial_curl": 33,
        }
        and "does not yet provide the acceleration-independent remainder" in item["scope"]
        for item in result["certificates"]
    )


def test_first_order_campaign_rejects_candidate_and_prerequisite_corruption() -> None:
    symmetrizer = _load(SYMMETRIZER_PATH)
    moser = _load(MOSER_PATH)
    config = _load(CONFIG_PATH)
    corrupted = json.loads(json.dumps(moser))
    corrupted["certificates"][0]["candidate_id"] = "corrupted-candidate"
    result = run_quartic_first_order_reduction_campaign(
        symmetrizer, corrupted, config
    )
    assert result["status"] == "reject"
    assert "campaign candidate sets do not match" in result["errors"]

    missing = json.loads(json.dumps(moser))
    missing["status"] = "reject"
    result = run_quartic_first_order_reduction_campaign(symmetrizer, missing, config)
    assert result["status"] == "reject"
    assert "Moser-coefficient campaign prerequisite failed" in result["errors"]
