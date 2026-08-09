from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.flrw_background import certify_flrw_background

ROOT = Path(__file__).resolve().parents[1]
IR_PATH = ROOT / "runs/physics-language/horndeski-l2-l4-polynomial-ir.json"
CONFIG_PATH = ROOT / "configs/backgrounds/canonical_scalar_stiff_interval.json"


def _inputs() -> tuple[dict, dict]:
    return json.loads(IR_PATH.read_text()), json.loads(CONFIG_PATH.read_text())


def test_canonical_scalar_stiff_flrw_is_interval_certified() -> None:
    ir, config = _inputs()
    result = certify_flrw_background(ir, config)
    assert result["status"] == "pass_interval_certified", result
    assert result["errors"] == []
    assert result["time"]["steps"] == 40
    assert result["analytic_reference"]["passed"]
    assert all(result["analytic_reference"]["contained"].values())
    uniform = result["uniform_certificate"]
    assert uniform["constraint_max_abs_enclosure"] <= uniform["constraint_tolerance"]
    assert uniform["evolution_determinant_min_abs"] > config["determinant_floor"]
    assert uniform["Theta_min_abs"] > config["health_margin"]
    assert all(
        value > config["health_margin"]
        for value in uniform["health_lower_bounds"].values()
    )
    formulation = result["formulation_certificate"]
    assert formulation["status"] == (
        "pass_generalized_harmonic_kessence_on_certified_trajectory"
    )
    assert formulation["route"] == "generalized_harmonic_kessence"
    assert all(
        value > config["health_margin"]
        for value in formulation["uniform_kessence_health_lower_bounds"].values()
    )
    assert "kessence_energy_density" in formulation[
        "uniform_kessence_health_lower_bounds"
    ]


def test_off_constraint_initial_state_is_rejected() -> None:
    ir, config = _inputs()
    config = copy.deepcopy(config)
    config["initial_state"]["x"] = "0.031"
    result = certify_flrw_background(ir, config)
    assert result["status"] == "reject"
    assert "initial energy-constraint" in " ".join(result["errors"])


def test_singular_evolution_matrix_is_rejected() -> None:
    ir, config = _inputs()
    ir = copy.deepcopy(ir)
    ir["compiled_flrw_background_system"]["evolution_matrix"] = (
        "Matrix([[1, 1], [1, 1]])"
    )
    result = certify_flrw_background(ir, config)
    assert result["status"] == "reject"
    assert "determinant interval" in " ".join(result["errors"])


def test_tensor_ghost_corruption_is_rejected() -> None:
    ir, config = _inputs()
    ir = copy.deepcopy(ir)
    ir["compiled_tensor_G_T"] = "-1"
    result = certify_flrw_background(ir, config)
    assert result["status"] == "reject"
    assert "G_T interval" in " ".join(result["errors"])


def test_negative_kessence_energy_on_trajectory_is_rejected() -> None:
    ir, config = _inputs()
    ir = copy.deepcopy(ir)
    ir["compiled_kessence_homogeneous_energy_density"] = "-x"
    result = certify_flrw_background(ir, config)
    assert result["status"] == "reject"
    assert "kessence_energy_density interval" in " ".join(result["errors"])


def test_g3_only_structure_is_routed_to_conditional_cubic_bssn() -> None:
    ir, config = _inputs()
    ir = copy.deepcopy(ir)
    ir["formulation_classification"]["canonical_G3"] = "x"
    result = certify_flrw_background(ir, config)
    assert result["status"] == "pass_interval_certified", result
    formulation = result["formulation_certificate"]
    assert formulation["route"] == "modified_harmonic_uniform_bound_required"
    assert formulation["status"] == (
        "unresolved_cubic_bssn_uniform_inhomogeneous_weak_field_bounds_required"
    )
    assert formulation["proof_route"] == "cubic_horndeski_bssn_weak_field"
    assert formulation["uniform_kessence_health_lower_bounds"] == {}
    diagnostic = formulation["cubic_bssn_homogeneous_diagnostic"]
    assert len(diagnostic["derivative_ratio_upper_bounds"]) == 15
    assert diagnostic["scalar_slicing_cone_gap_min_abs"] > config["health_margin"]


def test_interval_certificate_is_deterministic() -> None:
    ir, config = _inputs()
    first = certify_flrw_background(ir, config)
    second = certify_flrw_background(ir, config)
    assert first["content_sha256"] == second["content_sha256"]
