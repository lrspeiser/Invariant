from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item6_thermodynamic_state as item6

ROOT = Path(__file__).resolve().parents[1]


def test_config_binds_item5_and_forbids_mass_shortcuts() -> None:
    config = item6.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 6
    assert config["predecessor"]["required_decision"].endswith("ADVANCE_ITEM6")
    assert config["authorization"]["hecs_mass_profiles_allowed"] is False
    assert config["authorization"]["caustic_or_nfw_mass_allowed_as_predictor"] is False
    assert config["authorization"]["lensing_mass_allowed_as_predictor"] is False
    assert config["prefreeze_audit"]["hecs_velocity_response_values_read"] == 0


def test_sample_is_response_blind_and_balanced() -> None:
    config = item6.load_config(ROOT)
    manifest = item6.build_sample_manifest(ROOT)
    item6.validate_sample_manifest(manifest, config)
    assert manifest["counts"] == {
        "quality_passing_candidates": 28,
        "exploration": 20,
        "reserved_confirmation": 8,
        "exploration_cool_core": 7,
        "confirmation_cool_core": 3,
    }
    folds = [row["outer_fold"] for row in manifest["objects"] if row["role"] == "exploration"]
    assert [folds.count(index) for index in range(5)] == [4, 4, 4, 4, 4]
    assert manifest["prefreeze_boundary"]["hecs_velocity_response_values_read"] == 0
    assert manifest["prefreeze_boundary"]["reserved_confirmation_predictors_blinded"] is False
    assert manifest["prefreeze_boundary"]["reserved_confirmation_velocity_responses_blinded"]


def test_feature_builder_is_target_blind_and_finite() -> None:
    signature = inspect.signature(item6.measure_thermodynamic_features)
    assert "sigma" not in " ".join(signature.parameters)
    features = item6.measure_thermodynamic_features(
        temperature_kev=7.41,
        core_entropy_kev_cm2=29.8,
        entropy_100_kev_cm2=158.2,
        entropy_slope=0.82,
        xray_luminosity_1e43_erg_s=3.71,
        redshift=0.1412,
        member_count=116,
    )
    assert set(features) == set(item6.FEATURE_NAMES)
    assert all(np.isfinite(value) for value in features.values())
    assert features["cooling_squared"] >= 0


def test_creativity_boundary_requires_interactions_beyond_known_main_effects() -> None:
    config = item6.load_config(ROOT)
    models = {row["id"]: row for row in config["model_families"]}
    assert models["known_temperature_fixed"]["qualifying"] is False
    assert models["temperature_luminosity_nuisance"]["qualifying"] is False
    assert models["entropy_main_effects"]["qualifying"] is False
    assert models["entropy_gradient_coupling"]["qualifying"] is True
    assert models["cooling_state_coupling"]["qualifying"] is True
    assert models["all_thermodynamic_phase"]["qualifying"] is True


def test_exact_metadata_query_excludes_velocity_and_mass_products() -> None:
    url = item6._query_url(
        "J/ApJ/767/15/table1",
        columns=["Name", "RAJ2000", "DEJ2000", "z", "LX", "Cat", "Nm"],
        constraint_name="Name",
        constraint_value="A1201",
    )
    assert "sig" not in url
    assert "Mass" not in url
    assert "M200" not in url
    assert url.count("-out=") == 7


def test_synthetic_source_parsers_respect_frozen_schemas() -> None:
    temperatures = item6.parse_temperature_payload(
        b"Abell 1201\t4216\t0.1688\t5.61\tb\n", accept_name="Abell 1201"
    )
    entropy = item6.parse_entropy_payload(
        b"Abell 1201\textr\t12\t39.2\t2.0\t200.4\t5.0\t1.20\t0.05\t9.0e-01\n"
        b"Abell 1201\t\t\t0.0\t\t210.0\t5.0\t1.0\t0.1\t1.0e-03\n",
        accept_name="Abell 1201",
    )
    metadata = item6.parse_hecs_metadata_payload(
        b"A1201\t168.2270\t13.4350\t0.1671\t1.79\tBCS\t165\n",
        hecs_name="A1201",
    )
    response = item6.parse_hecs_response_payload(b"A1201\t780\t60\t50\n", hecs_name="A1201")
    assert temperatures[0]["temperature_kev"] == 5.61
    assert len(entropy) == 1
    assert entropy[0]["core_entropy_kev_cm2"] == 39.2
    assert metadata["member_count"] == 165
    assert response["sigma_km_s"] == 780


def test_response_acquisition_is_impossible_before_freeze_binding() -> None:
    if item6.SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(item6.GravityItem6ThermodynamicStateError, match="not bound"):
            item6.acquire_exploration(ROOT)


def test_stored_sample_matches_builder_after_it_is_written() -> None:
    config = item6.load_config(ROOT)
    path = ROOT / config["sample_manifest_output"]
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == item6.build_sample_manifest(ROOT)
