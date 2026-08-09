from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

from sigma_theory_compiler.quartic_geometric_jet_campaign import (
    geometric_state_to_jet_control,
    run_quartic_geometric_jet_campaign,
    state_to_covariant_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
FIRST_ORDER_PATH = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-first-order-reduction-campaign"
    / "campaign.json"
)
CONFIG_PATH = ROOT / "configs" / "backgrounds" / "quartic_geometric_jet_campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_state_to_covariant_jet_controls_include_lower_order_terms() -> None:
    passed, evidence = geometric_state_to_jet_control()
    assert passed
    assert evidence["state"]["U_dimension"] == 55
    assert evidence["curvilinear_flat_control"]["connection_nonzero"]
    assert evidence["curvilinear_flat_control"]["riemann_zero"]
    assert evidence["off_diagonal_basis_control"] == {
        "metric": "-dt^2+(dx+y dy)^2+dy^2+dz^2",
        "q_12": "sqrt(2)*y",
        "metric_roundtrip_residual_zero": True,
        "riemann_zero": True,
    }
    assert evidence["curved_control"]["einstein_residuals"] == ["0"] * 16
    assert all(
        item["rejected"] for item in evidence["negative_controls"].values()
    )


def test_state_adapter_rejects_wrong_dimensions() -> None:
    try:
        state_to_covariant_geometry([sp.Integer(0)] * 54, [[sp.Integer(0)] * 55] * 4)
    except ValueError as error:
        assert "U[55]" in str(error)
    else:
        raise AssertionError("wrong state dimension was accepted")


def test_all_candidates_bind_to_exact_nonlinear_geometric_map() -> None:
    result = run_quartic_geometric_jet_campaign(
        _load(FIRST_ORDER_PATH), _load(CONFIG_PATH)
    )
    assert result["status"] == (
        "pass_all_12_exact_nonlinear_geometric_state_to_jet_maps"
    )
    assert result["counts"] == {
        "selected": 12,
        "geometric_state_to_jet_maps_passed": 12,
        "rejected": 0,
    }
    assert all(
        certificate["state_dimension"] == 55
        and certificate["resolved_predecessor_gate"]
        == "incidence_defined_nonlinear_formula_map_unresolved"
        and certificate["remaining_gate"]
        == "quartic_gauge_fixed_nonlinear_evolution_source"
        for certificate in result["certificates"]
    )


def test_geometric_campaign_rejects_corrupted_prerequisite() -> None:
    first_order = _load(FIRST_ORDER_PATH)
    config = _load(CONFIG_PATH)
    corrupted = json.loads(json.dumps(first_order))
    corrupted["status"] = "reject"
    result = run_quartic_geometric_jet_campaign(corrupted, config)
    assert result["status"] == "reject"
    assert "first-order campaign prerequisite failed" in result["errors"]
