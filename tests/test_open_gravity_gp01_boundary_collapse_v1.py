from __future__ import annotations

import copy
import math

import pytest

from sigma_theory_compiler import open_gravity_gp01_boundary_collapse_v1 as collapse


def test_target_is_bounded_by_equilibrium_gain() -> None:
    for y in (1.0e-4, 0.01, 0.1, 1.0, 10.0, 100.0):
        gamma_l = collapse.equilibrium_log_gain(y, 1)
        target = collapse.bounded_target(gamma_l, window=0.7, gamma_max=math.log(8.0))
        assert 0.0 <= target <= gamma_l


def test_exact_error_decomposition_and_bound() -> None:
    gamma_l = collapse.equilibrium_log_gain(0.03, 2)
    window = 0.63
    gamma_max = math.log(8.0)
    target = collapse.bounded_target(gamma_l, window=window, gamma_max=gamma_max)
    error = gamma_l - target
    decomposition = (1.0 - window) * gamma_l + window * (
        gamma_l - gamma_max * math.tanh(gamma_l / gamma_max)
    )
    assert error == pytest.approx(decomposition, abs=2.0e-16)
    assert (
        error
        <= collapse.analytic_error_bound(gamma_l, window=window, gamma_max=gamma_max) + 2.0e-16
    )


def test_full_probe_suite_and_limit() -> None:
    suite = collapse.run_suite(collapse.load_config())
    assert suite["probe_cases"] == 3 * 6 * 6 * 5 * 5 * 2 * 2
    assert suite["monotone_gamma_max"]
    assert suite["monotone_environment_thresholds"]
    assert suite["maximum_limit_relative_error"] < 1.0e-5


def test_empirical_loss_and_broader_family_are_both_retained() -> None:
    receipt = collapse.build_receipt()
    evidence = receipt["empirical_boundary_evidence"]
    assert evidence["elliptic_beats_equilibrium_on_objects"] == 0
    assert evidence["elliptic_to_equilibrium_robust_loss_ratio"] > 1.0
    assert receipt["decision"]["boundary_extension"].startswith("STOP_AS_NOVELTY")
    assert receipt["decision"]["broader_gp01_history_family"] == "ACTIVE"


def test_config_mutations_fail() -> None:
    config = collapse.load_config()
    mutated = copy.deepcopy(config)
    mutated["decision"]["broader_gp01_history_family"] = "ELIMINATED"
    with pytest.raises(collapse.BoundaryCollapseError):
        collapse.validate_config(mutated)


def test_zero_scientific_access() -> None:
    assert set(collapse.build_receipt()["access_accounting"].values()) == {0}


def test_receipt_round_trip(tmp_path, monkeypatch) -> None:
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(collapse, "OUTPUT_PATH", output)
    assert collapse.write_receipt() == "CREATED"
    collapse.validate_receipt()
    assert collapse.write_receipt() == "EXISTING_IDENTICAL"
