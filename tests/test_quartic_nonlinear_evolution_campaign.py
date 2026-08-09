from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.quartic_nonlinear_evolution_campaign import (
    nonlinear_evolution_source_control,
    run_quartic_nonlinear_evolution_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
GEOMETRIC_PATH = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-geometric-jet-campaign"
    / "campaign.json"
)
CONFIG_PATH = (
    ROOT / "configs" / "backgrounds" / "quartic_nonlinear_evolution_campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_nonlinear_source_matches_independent_principal_and_solves() -> None:
    passed, evidence = nonlinear_evolution_source_control()
    assert passed
    assert evidence["equation_count"] == 11
    assert evidence["acceleration_count"] == 11
    assert evidence["time_acceleration_affine_residual_zero"]
    assert evidence["independent_principal_time_block_residual_zero"]
    assert evidence["nonzero_acceleration_independent_remainder"]
    assert evidence["known_answer_reductions"] == {
        "alpha_0_c20_0": {
            "theory": "Einstein-Hilbert plus canonical scalar",
            "metric_residual_zero": True,
            "scalar_box_residual": "0",
        },
        "constant_scalar": {
            "theory": "pure Einstein-Hilbert",
            "metric_residual_zero": True,
            "scalar_residual": "0",
        },
    }
    assert evidence["sample_solution"]["solution_residual_zero"]
    assert evidence["curvilinear_reference_connection_control"] == {
        "metric": "diag(-1,1,r^2,1)",
        "physical_connection_nonzero": True,
        "Delta_Gamma_zero_with_matching_flat_reference": True,
        "gauge_completion_zero": True,
        "omitted_reference_connection_nonzero": True,
    }
    assert all(
        item["rejected"] for item in evidence["negative_controls"].values()
    )


def test_all_candidates_solve_nonzero_acceleration_independent_remainder_inside_certified_box() -> None:
    result = run_quartic_nonlinear_evolution_campaign(
        _load(GEOMETRIC_PATH), _load(CONFIG_PATH)
    )
    assert result["status"] == (
        "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
    )
    assert result["counts"] == {
        "selected": 12,
        "nonlinear_time_acceleration_eliminations_passed": 12,
        "rejected": 0,
    }
    assert all(
        item["acceleration_solution_residual_zero"]
        and item["nonzero_acceleration_independent_remainder"]
        and item["maximum_solved_jet_component_numeric"] < 2e-10
        and item["remaining_gate"]
        == "nonlinear_source_symmetrizer_derivative_bounds_and_pde_bootstrap"
        for item in result["certificates"]
    )


def test_nonlinear_campaign_rejects_corrupted_prerequisite() -> None:
    geometric = _load(GEOMETRIC_PATH)
    corrupted = json.loads(json.dumps(geometric))
    corrupted["status"] = "reject"
    result = run_quartic_nonlinear_evolution_campaign(
        corrupted, _load(CONFIG_PATH)
    )
    assert result["status"] == "reject"
    assert "geometric-jet campaign prerequisite failed" in result["errors"]
