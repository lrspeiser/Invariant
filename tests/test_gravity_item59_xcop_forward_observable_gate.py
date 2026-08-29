from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    CONFIG_PATH,
    GravityItem59Error,
    _cumulative_mass,
    _law_acceleration,
    _predict_variant,
    _score_predictions,
    _split_order,
    enumerate_variants,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def _synthetic_packet() -> dict[str, object]:
    density_radius = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    return {
        "cluster": "SYNTHETIC",
        "r500_kpc": 100.0,
        "density_radius_kpc": density_radius,
        "ne_cm3": np.asarray([0.020, 0.014, 0.010, 0.007, 0.005, 0.0035]),
        "ne_error_low_cm3": np.full(6, 0.0002),
        "ne_error_high_cm3": np.full(6, 0.0003),
        "stellar": None,
        "anchor": {
            "index": 5,
            "radius_kpc": 60.0,
            "pressure_kev_cm3": 0.001,
            "error_kev_cm3": 0.0001,
        },
        "rows": [
            {
                "row_id": "SYNTHETIC:pressure:0",
                "cluster": "SYNTHETIC",
                "observable": "pressure",
                "radius_kpc": 20.0,
                "observed": 0.003,
                "error": 0.0003,
                "split": "confirmation",
            },
            {
                "row_id": "SYNTHETIC:pressure:1",
                "cluster": "SYNTHETIC",
                "observable": "pressure",
                "radius_kpc": 40.0,
                "observed": 0.002,
                "error": 0.0002,
                "split": "confirmation",
            },
            {
                "row_id": "SYNTHETIC:temperature:0",
                "cluster": "SYNTHETIC",
                "observable": "temperature",
                "radius_kpc": 30.0,
                "observed": 5.0,
                "error": 0.5,
                "split": "confirmation",
            },
            {
                "row_id": "SYNTHETIC:temperature:1",
                "cluster": "SYNTHETIC",
                "observable": "temperature",
                "radius_kpc": 50.0,
                "observed": 4.0,
                "error": 0.4,
                "split": "confirmation",
            },
        ],
    }


def _newtonian_variant() -> dict[str, object]:
    return {
        "variant_id": "synthetic-newtonian",
        "family_id": "newtonian_baryons",
        "origin_label": "known_physical_baseline",
        "qualifying": False,
        "parameters": {},
        "nuisances": {
            "outer_nonthermal_fraction": 0.0,
            "published_stellar_mass_scale": 1.0,
            "missing_stellar_to_gas_mass_ratio": 0.1,
            "xray_temperature_cross_calibration": 1.0,
        },
    }


def test_unbound_scientific_contract_is_valid_before_freeze() -> None:
    config = _config()
    validate_config(ROOT, config, require_bound=False)
    assert config["population"]["confirmation_response_rows_allowed_before_freeze"] is False
    assert config["population"]["inferred_total_mass_rows_allowed"] == 0
    assert config["counterexample_policy"]["single_counterexample_terminal"] is False


def test_total_mass_and_single_counterexample_tampers_are_rejected() -> None:
    total_mass = deepcopy(_config())
    total_mass["observable_contract"]["total_mass_used_anywhere"] = True
    with pytest.raises(GravityItem59Error, match="total-mass"):
        validate_config(ROOT, total_mass, require_bound=False)
    singleton = deepcopy(_config())
    singleton["counterexample_policy"]["single_counterexample_terminal"] = True
    with pytest.raises(GravityItem59Error, match="over-pruning"):
        validate_config(ROOT, singleton, require_bound=False)


def test_variant_enumeration_is_complete_unique_and_mostly_creative() -> None:
    variants = enumerate_variants(_config())
    assert len(variants) == 2025
    assert len({row["variant_id"] for row in variants}) == 2025
    assert sum(row["qualifying"] for row in variants) == 23 * 81


def test_radial_split_order_is_deterministic_and_response_blind() -> None:
    first = _split_order("A85", "pressure", 3, "salt")
    second = _split_order("A85", "pressure", 3, "salt")
    assert first == second
    assert first != _split_order("A85", "pressure", 4, "salt")


def test_cumulative_mass_recovers_uniform_sphere_endpoint() -> None:
    radius = np.linspace(1.0, 10.0, 1000)
    density = np.full_like(radius, 2.0)
    mass = _cumulative_mass(radius, density)
    expected = 4.0 * np.pi * density[-1] * radius[-1] ** 3 / 3.0
    assert mass[-1] == pytest.approx(expected, rel=2.0e-6)


@pytest.mark.parametrize(
    ("family", "parameters"),
    [
        ("newtonian_baryons", {}),
        ("empirical_rar", {}),
        ("cross_scale_boundary", {"beta": 2.0}),
        ("distance_running_gravity", {"amplitude": 2.0, "power": 1.0}),
        ("qed_like_screened_coupling", {"amplitude": 2.0, "power": 1.0}),
        (
            "interior_resonance_equalization",
            {"amplitude": 1.0, "log_radius_scale": 0.5},
        ),
    ],
)
def test_all_law_families_produce_positive_acceleration(
    family: str, parameters: dict[str, float]
) -> None:
    radius = np.geomspace(10.0, 1000.0, 50)
    gbar = 3.0e-10 * (radius / radius[0]) ** -1.2
    acceleration = _law_acceleration(family, parameters, radius, 1000.0, gbar, _config())
    assert acceleration.shape == radius.shape
    assert np.all(np.isfinite(acceleration))
    assert np.all(acceleration > 0.0)


def test_hydrostatic_forward_model_predicts_pressure_and_temperature() -> None:
    packet = _synthetic_packet()
    predictions = _predict_variant(packet, _newtonian_variant(), _config())
    assert predictions["SYNTHETIC:pressure:0"] > predictions["SYNTHETIC:pressure:1"]
    assert all(np.isfinite(value) and value > 0.0 for value in predictions.values())
    score = _score_predictions([packet], predictions, "confirmation", _config())
    assert score["rows"] == 4
    assert set(score["by_observable"]) == {"pressure", "temperature"}
    assert set(score["by_cluster_observable"]) == {
        "SYNTHETIC:pressure",
        "SYNTHETIC:temperature",
    }
