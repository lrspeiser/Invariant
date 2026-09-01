from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_differential_propagation_kernel_v1 as lane

ROOT = Path(__file__).resolve().parents[1]


def _config_unsealed() -> dict[str, object]:
    return json.loads((ROOT / lane.CONFIG_PATH).read_text(encoding="utf-8"))


def test_analytic_controls_prove_static_limit_energy_and_front_boundary() -> None:
    controls = lane.analytic_controls()
    assert len(controls) == 8
    assert all(item["passed"] for item in controls.values())
    assert controls["A01_STATIC_POISSON_INDEPENDENT_OF_CG"]["residual"] == "0"
    assert controls["A02_STATIC_DAMPING_DROPS_OUT"]["residual"] == "0"
    assert controls["A04_CONSERVATIVE_ENERGY_IDENTITY"]["residual"] == "0"
    assert controls["A08_QUARTIC_PHASE_SPEED_UNBOUNDED"]["residual"] == "oo"


def test_luminal_massless_radiative_control_is_exact() -> None:
    for omega in (0.2, 0.7, 1.4, 2.3):
        wave_number = lane.radiative_wavenumber(
            omega, c_g=1.0, gamma=0.0, mu=0.0, zeta=0.0, k_star=3.0
        )
        assert wave_number.real == pytest.approx(omega, abs=2.0e-15)
        assert wave_number.imag == pytest.approx(0.0, abs=2.0e-15)


def test_passive_damping_attenuates_and_never_amplifies() -> None:
    for omega in (0.5, 1.0, 2.0):
        wave_number = lane.radiative_wavenumber(
            omega, c_g=0.9, gamma=0.03, mu=0.02, zeta=0.0, k_star=2.0
        )
        assert wave_number.imag > 0.0
        near = abs(
            lane.propagation_transfer(
                omega,
                2.0,
                c_g=0.9,
                gamma=0.03,
                mu=0.02,
                zeta=0.0,
                k_star=2.0,
            )
            * 2.0
        )
        far = abs(
            lane.propagation_transfer(
                omega,
                8.0,
                c_g=0.9,
                gamma=0.03,
                mu=0.02,
                zeta=0.0,
                k_star=2.0,
            )
            * 8.0
        )
        assert far < near < 1.0


@pytest.mark.parametrize(
    "parameters",
    [
        {"omega": 1.0, "c_g": 0.0, "gamma": 0.0, "mu": 0.0, "zeta": 0.0, "k_star": 1.0},
        {"omega": 1.0, "c_g": 1.0, "gamma": -0.1, "mu": 0.0, "zeta": 0.0, "k_star": 1.0},
        {"omega": 1.0, "c_g": 1.0, "gamma": 0.0, "mu": -0.1, "zeta": 0.0, "k_star": 1.0},
        {"omega": 1.0, "c_g": 1.0, "gamma": 0.0, "mu": 0.0, "zeta": -0.1, "k_star": 1.0},
    ],
)
def test_active_or_ill_posed_parameters_fail_closed(parameters: dict[str, float]) -> None:
    with pytest.raises(lane.DifferentialPropagationError):
        lane.radiative_wavenumber(**parameters)


def test_target_free_injection_recovery_is_exact_without_observations() -> None:
    recoveries = lane.injection_recovery(_config_unsealed())
    assert [row["injection_id"] for row in recoveries] == [
        "INJ_CONSERVATIVE_DISPERSIVE",
        "INJ_ATTENUATING_SUPERLUMINAL",
    ]
    assert all(row["exact_recovery"] for row in recoveries)
    assert all(row["complex_residual"] < 1.0e-25 for row in recoveries)
    assert all(row["observational_targets_used"] == 0 for row in recoveries)
    assert all(row["candidate_count"] >= 36 for row in recoveries)


def test_branch_falsifiers_retain_unconventional_sectors() -> None:
    rows = lane.classify_branches(_config_unsealed())
    by_id = {row["id"]: row for row in rows}
    assert len(rows) == 8
    assert by_id["UNIVERSAL_SLOW_1E12"]["decision"] == (
        "FALSIFIED_IF_UNIVERSAL_RADIATIVE_BY_GW170817_SPEED"
    )
    assert by_id["UNIVERSAL_FAST_1E12"]["decision"] == (
        "FALSIFIED_IF_UNIVERSAL_RADIATIVE_BY_GW170817_SPEED"
    )
    assert by_id["UNIVERSAL_MASS_1E22"]["decision"] == (
        "FALSIFIED_IF_TESTED_TENSOR_MASS_BY_2026_GWTC3_BOUND"
    )
    assert by_id["SCREENED_SLOW_MEMORY"]["gw170817_speed_pass"] is None
    assert by_id["SCREENED_SLOW_MEMORY"]["decision"].startswith("RETAINED_NONUNIVERSAL")
    assert by_id["LOSSY_WITHOUT_BATH"]["closed_field_energy_conservation_pass"] is False
    assert by_id["LOSSY_WITHOUT_BATH"]["decision"].startswith("RETAINED_OPEN_SYSTEM")
    assert by_id["QUARTIC_EFT"]["finite_front_claim_available"] is False
    assert by_id["QUARTIC_EFT"]["decision"].startswith("RETAINED_EFT_BRANCH")


def test_gw_solar_and_binary_applicability_is_not_overclaimed() -> None:
    config = _config_unsealed()
    rows = lane.classify_branches(config)
    assert all(row["solar_static_limit"] == "PASS_EXACT_POISSON" for row in rows)
    assert all(row["solar_moving_body"].startswith("NOT_DERIVED") for row in rows)
    assert all(row["binary_radiation_reaction"].startswith("NOT_DERIVED") for row in rows)
    benchmarks = lane.benchmark_map(config)
    assert benchmarks["B1913_ORBITAL_DECAY"]["value"] == {
        "observed_to_gr_ratio": 0.9983,
        "sigma": 0.0016,
    }
    assert benchmarks["DOUBLE_PULSAR_QUADRUPOLE_DAMPING"]["value"] == {
        "fractional_validation_95pct": 0.00013
    }


def test_exact_public_product_and_acquisition_receipt_are_frozen() -> None:
    preflight = _config_unsealed()["public_data_preflight"]
    assert preflight["state"] == "SOURCE_READY_NOT_OPENED_BY_BUILDER"
    assert preflight["dataset_doi"] == "10.7935/K5B8566F"
    assert preflight["event_uid"] == "GW170817-v1"
    assert preflight["event_gps"] == pytest.approx(1187008882.4)
    assert all(detector in preflight["exact_product"] for detector in ("H1", "L1", "V1"))
    assert len(preflight["required_acquisition_receipt"]) == 6
    assert "glitch handling" in preflight["required_acquisition_receipt"][4]


def test_receipt_is_deterministic_and_preserves_strongest_counterexample() -> None:
    first = lane.build_receipt(ROOT)
    second = lane.build_receipt(ROOT)
    assert first == second
    assert first["checks_passed"] == first["checks_total"] == 16
    assert all(first["checks"].values())
    assert first["strongest_counterexample"]["result"] == (
        "A gravity speed different from light does not by itself increase the force "
        "of a stationary source."
    )
    assert first["claim_boundary"]["historical_novelty_established"] is False
    assert first["claim_boundary"]["publication_ready"] is False
    assert first["access_ledger"] == {
        "observational_files_opened": 0,
        "observational_rows_read": 0,
        "real_scores_computed": 0,
        "network_calls_by_builder": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }
    assert first["content_sha256"] == lane._self_hash(first)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("claim_boundary", "historical_novelty_established", True),
        ("claim_boundary", "real_observational_rows_scored", True),
        ("claim_boundary", "solar_moving_body_prediction", True),
        ("claim_boundary", "binary_radiation_reaction_prediction", True),
        ("claim_boundary", "publication_ready", True),
        ("access_ledger", "observational_rows_read", 1),
        ("access_ledger", "paid_calls", 1),
    ],
)
def test_claim_and_access_mutations_fail_closed(section: str, key: str, value: object) -> None:
    config = copy.deepcopy(_config_unsealed())
    config[section][key] = value
    with pytest.raises(lane.DifferentialPropagationError):
        lane.validate_config(config)


def test_rehashed_receipt_overclaim_fails_closed() -> None:
    config = lane.load_config(ROOT)
    receipt = lane.build_receipt(ROOT)
    receipt["claim_boundary"]["covariant_tensor_completion"] = True
    receipt["content_sha256"] = lane._self_hash(receipt)
    with pytest.raises(lane.DifferentialPropagationError, match="claims changed"):
        lane.validate_receipt(receipt, config)


def test_atomic_write_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"content_sha256": "one"}
    assert lane._atomic_no_clobber(path, value) == "CREATED"
    assert lane._atomic_no_clobber(path, value) == "EXISTING_IDENTICAL"
    with pytest.raises(lane.DifferentialPropagationError, match="refusing to replace"):
        lane._atomic_no_clobber(path, {"content_sha256": "two"})


def test_phase_curvature_is_nonzero_for_massive_or_quartic_branches() -> None:
    frequencies = [0.9, 1.2, 1.7]
    phases = [
        lane.radiative_wavenumber(
            frequency,
            c_g=1.0,
            gamma=0.0,
            mu=0.08,
            zeta=0.2,
            k_star=2.5,
        ).real
        for frequency in frequencies
    ]
    slope_low = (phases[1] - phases[0]) / (frequencies[1] - frequencies[0])
    slope_high = (phases[2] - phases[1]) / (frequencies[2] - frequencies[1])
    assert not math.isclose(slope_low, slope_high, rel_tol=1.0e-6, abs_tol=1.0e-6)
