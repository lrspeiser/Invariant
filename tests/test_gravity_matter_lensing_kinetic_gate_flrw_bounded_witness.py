from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_kinetic_gate_flrw_bounded_witness as witness,
)

ROOT = Path(__file__).resolve().parents[1]


def test_gate_threshold_and_exact_factor() -> None:
    below = witness.gate_terms(math.sqrt(0.3), 1.0)
    above = witness.gate_terms(math.sqrt(0.34), 1.0)
    assert below["u"] == pytest.approx(0.3)
    assert below["M"] > 0.0
    assert above["M"] < 0.0
    direct = below["Z"] * below["H_gate"] - 4.0 * math.sqrt(0.3) * below["Z_X"] ** 2
    assert direct == pytest.approx(below["M"])


def test_initial_current_reconstruction() -> None:
    x_initial = math.sqrt(0.3)
    y_initial = 0.02
    j_phi, j_chi = witness.initial_currents(x_initial, y_initial, 1.0)
    x_value, y_value = witness.solve_invariants(1.0, j_phi, j_chi, 1.0, 180)
    assert x_value == pytest.approx(x_initial, rel=1e-14)
    assert y_value == pytest.approx(y_initial, rel=1e-14)


def test_expansion_reduces_both_invariants() -> None:
    x_initial = math.sqrt(0.3)
    j_phi, j_chi = witness.initial_currents(x_initial, 0.02, 1.0)
    x_late, y_late = witness.solve_invariants(10.0, j_phi, j_chi, 1.0, 180)
    assert x_late < x_initial
    assert y_late < 0.02
    assert witness.gate_terms(x_late, 1.0)["u"] < 0.3


def test_full_trajectory_is_on_shell_and_healthy() -> None:
    config = witness.load_config(ROOT)
    trajectory = witness.build_trajectory(config)
    assert len(trajectory) == 81
    assert trajectory[0]["u"] == pytest.approx(0.3)
    assert trajectory[-1]["a"] == pytest.approx(10.0)
    assert all(trajectory[index]["u"] > trajectory[index + 1]["u"] for index in range(80))
    assert max(row["u"] for row in trajectory) < 1.0 / 3.0
    assert min(row["M"] for row in trajectory) > 0.0
    assert min(row["gradient_min_eigenvalue"] for row in trajectory) > 0.0
    assert min(row["kinetic_min_eigenvalue"] for row in trajectory) > 0.0
    assert min(row["kinetic_determinant"] for row in trajectory) > 0.0
    assert min(row["sound_speed_squared_min"] for row in trajectory) > 0.0
    assert max(row["phi_eom_residual"] for row in trajectory) < 1e-10
    assert max(row["chi_eom_residual"] for row in trajectory) < 1e-10
    assert max(row["continuity_relative_residual"] for row in trajectory) < 1e-10
    assert max(row["friedmann_absolute_residual"] for row in trajectory) < 1e-10
    assert max(row["raychaudhuri_absolute_residual"] for row in trajectory) < 1e-10


def test_stress_and_raychaudhuri_identity_at_checkpoint() -> None:
    config = witness.load_config(ROOT)
    row = witness.build_trajectory(config)[20]
    rho_plus_p = row["energy_density"] + row["pressure"]
    expected = 2.0 * row["X"] * row["C"] + 2.0 * row["Y"] * row["Z"]
    assert rho_plus_p == pytest.approx(expected)
    assert row["equation_of_state"] > 0.0
    assert row["equation_of_state"] < 1.0


def test_symbolic_derivation_routes_pass() -> None:
    checks = witness.symbolic_checks()
    assert checks == {
        "W02_COVARIANT_TO_FLRW_STRESS": True,
        "W03_SCALAR_CURRENT_REDUCTION": True,
        "W04_EXACT_GATE_DERIVATIVES": True,
        "W05_EXACT_M_FACTOR": True,
        "W13_KINETIC_MATRIX_POSITIVE": True,
        "W19_UNIQUE_POSITIVE_CURRENT_BRANCH": True,
        "W20_INDEPENDENT_QUADRATIC_HESSIAN": True,
        "W21_GENERAL_MULTIFIELD_ADM_MAPPING": True,
    }


@pytest.mark.parametrize(
    ("x_value", "beta"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (float("nan"), 1.0)],
)
def test_invalid_gate_inputs_fail_closed(x_value: float, beta: float) -> None:
    with pytest.raises(witness.KineticGateFlrwWitnessError):
        witness.gate_terms(x_value, beta)


def test_invalid_solver_surface_fails_closed() -> None:
    j_phi, j_chi = witness.initial_currents(math.sqrt(0.3), 0.02, 1.0)
    with pytest.raises(witness.KineticGateFlrwWitnessError):
        witness.solve_invariants(0.9, j_phi, j_chi, 1.0, 180)
    with pytest.raises(witness.KineticGateFlrwWitnessError):
        witness.solve_invariants(1.0, j_phi, j_chi, 1.0, 20)
    with pytest.raises(
        witness.KineticGateFlrwWitnessError,
        match="no root inside the bounded positive-mixing interval",
    ):
        witness.solve_invariants(1.0, 1.0e6, j_chi, 1.0, 180)


def test_receipt_is_deterministic_and_restrictive() -> None:
    first = witness.build_receipt(ROOT)
    second = witness.build_receipt(ROOT)
    assert first == second
    assert first["checks_passed"] == 21
    assert first["content_sha256"] == witness._self_hash(first)
    assert first["implementation_binding"]["module_sha256"] == witness._sha256_file(
        ROOT / first["implementation_binding"]["module_path"]
    )
    assert first["implementation_binding"]["test_sha256"] == witness._sha256_file(
        ROOT / first["implementation_binding"]["test_path"]
    )
    assert (
        first["status"]
        == "PASS_COUPLED_EINSTEIN_SCALAR_ON_SHELL_POSITIVE_BLOCK_WITNESS_SUPERLUMINAL_WARNING"
    )
    assert first["claim_boundary"]["coupled_metric_and_scalar_background_on_shell"] is True
    assert (
        first["claim_boundary"]["einstein_constrained_high_frequency_scalar_principal_block"]
        is True
    )
    assert first["claim_boundary"]["metric_cone_subluminality"] is False
    assert first["claim_boundary"]["unbounded_growing_gate_healthy"] is False
    assert first["claim_boundary"]["publication_ready"] is False
    assert first["witness"]["health_extrema"]["u_max"] < 1.0 / 3.0
    assert first["witness"]["health_extrema"]["sound_speed_squared_max"] > 1.0


def test_config_semantic_mutation_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(witness.load_config(ROOT))
    config["claim_boundary"]["publication_ready"] = True
    monkeypatch.setattr(witness, "_read_json", lambda _path: config)
    with pytest.raises(witness.KineticGateFlrwWitnessError, match="semantics changed"):
        witness.load_config(ROOT)


def test_predecessor_git_and_worktree_bytes_are_exact() -> None:
    config = witness.load_config(ROOT)
    for predecessor in config["predecessors"]:
        for role in ("config", "module", "test", "receipt"):
            relative = predecessor[f"{role}_path"]
            expected = predecessor[f"{role}_sha256"]
            assert witness._sha256_file(ROOT / relative) == expected
            assert (
                witness._sha256_bytes(witness._git_show(ROOT, predecessor["commit"], relative))
                == expected
            )
