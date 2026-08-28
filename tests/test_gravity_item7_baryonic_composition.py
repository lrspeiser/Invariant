from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item7_baryonic_composition as item7

ROOT = Path(__file__).resolve().parents[1]


def test_config_binds_item6_and_forbids_dynamical_shortcuts() -> None:
    config = item7.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 7
    assert config["predecessor"]["required_decision"].endswith("ADVANCE_ITEM7")
    assert config["authorization"]["dark_matter_or_dynamical_mass_allowed_as_predictor"] is False
    assert config["authorization"]["rotation_speed_or_curve_shape_allowed_as_predictor"] is False
    assert config["authorization"]["lensing_mass_allowed_as_predictor"] is False
    assert config["prefreeze_audit"]["rotation_velocity_values_read"] == 0


def test_sample_is_response_blind_balanced_and_sealed() -> None:
    config = item7.load_config(ROOT)
    manifest = item7.build_sample_manifest(ROOT)
    item7.validate_sample_manifest(manifest, config)
    assert manifest["counts"] == {
        "catalog_overlap": 67,
        "base_quality": 54,
        "radius_coverage_quality": 45,
        "exploration": 33,
        "reserved_confirmation": 12,
    }
    folds = [row["outer_fold"] for row in manifest["objects"] if row["role"] == "exploration"]
    assert sorted(folds.count(index) for index in range(5)) == [6, 6, 7, 7, 7]
    assert manifest["prefreeze_boundary"]["rotation_velocity_values_read"] == 0
    assert manifest["prefreeze_boundary"]["reserved_confirmation_rotation_responses_blinded"]


def test_feature_builder_is_target_blind_and_finite() -> None:
    signature = inspect.signature(item7.measure_composition_features)
    forbidden = " ".join(signature.parameters).lower()
    assert "velocity" not in forbidden
    assert "rotation" not in forbidden
    features = item7.measure_composition_features(
        log_mstar=10.4,
        log_mhi=9.7,
        log_lco=8.4,
        co_aperture_correction=1.3,
        stellar_effective_radius_kpc=3.9,
        stellar_scale_length_kpc=2.9,
        optical_radius_kpc=14.1,
        log_sfr=0.24,
        inclination_deg=42.0,
        co_covering_fraction=0.29,
        rco_over_r25=0.55,
    )
    assert set(features) == set(item7.FEATURE_NAMES)
    assert all(np.isfinite(value) for value in features.values())
    assert sum(features[name] for name in ("stellar_fraction", "atomic_fraction", "molecular_fraction")) == pytest.approx(1.0)
    assert 0.0 <= features["phase_entropy"] <= 1.0


def test_creativity_boundary_separates_known_scalings_from_interactions() -> None:
    models = {row["id"]: row for row in item7.load_config(ROOT)["model_families"]}
    assert models["btfr_fixed"]["qualifying"] is False
    assert models["newtonian_mass_size_fixed"]["qualifying"] is False
    assert models["phase_main_effects"]["qualifying"] is False
    assert models["phase_balance_coupling"]["qualifying"] is True
    assert models["gas_phase_geometry_coupling"]["qualifying"] is True


def test_predictor_query_cannot_request_rotation_response() -> None:
    config = item7.load_config(ROOT)
    source = config["sources"]["composition_table"]
    url = item7._query_url(
        source["catalog_id"],
        columns=source["allowed_columns"],
        constraint_name="Name",
        constraint_value="NGC0628",
    )
    assert "VRot" not in url
    assert "velocity" not in url.lower()
    assert url.count("-out=") == 9


def test_synthetic_source_parsers_and_interpolation() -> None:
    composition = item7.parse_composition_payload(
        b"NGC0628\t10.34\t14.1\t3.9\t2.9\t0.24\t8.41\t1.73\t9.70\n",
        galaxy="NGC0628",
    )
    metadata = item7.parse_metadata_payload(
        b"NGC0628\t28.9\t0.29\t0.00\t0.55\t43\n", galaxy="NGC0628"
    )
    curve = item7.parse_curve_payload(
        b"NGC0628\t1.0\t100.0\t5.0\t4.0\nNGC0628\t2.0\t140.0\t7.0\t6.0\n",
        galaxy="NGC0628",
    )
    interpolated = item7._interpolate_curve(curve, radius_kpc=1.5)
    assert composition["lstar_kpc"] == 2.9
    assert metadata["n_rotation_rows"] == 43
    assert interpolated == {
        "velocity_km_s": 120.0,
        "upper_error_km_s": 6.0,
        "lower_error_km_s": 5.0,
    }


def test_response_acquisition_is_impossible_before_freeze_binding() -> None:
    if item7.SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(item7.GravityItem7BaryonicCompositionError, match="not bound"):
            item7.acquire_exploration(ROOT)


def test_stored_sample_matches_builder_after_it_is_written() -> None:
    config = item7.load_config(ROOT)
    path = ROOT / config["sample_manifest_output"]
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == item7.build_sample_manifest(ROOT)


def test_source_manifest_keeps_confirmation_and_mass_shortcuts_closed() -> None:
    config = item7.load_config(ROOT)
    path = ROOT / config["source_manifest_output"]
    if not path.exists():
        pytest.skip("exploration source not acquired")
    source = json.loads(path.read_text(encoding="utf-8"))
    item7.validate_source_manifest(source, config)
    assert source["boundary"]["reserved_confirmation_primary_response_queries"] == 0
    assert source["boundary"]["dynamical_or_dark_mass_values_acquired"] == 0
    assert source["boundary"]["lensing_mass_values_acquired"] == 0


def test_receipt_replays_if_experiment_has_run() -> None:
    config = item7.load_config(ROOT)
    path = ROOT / config["output"]
    if not path.exists():
        pytest.skip("experiment not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert item7.build_receipt(ROOT) == stored
    item7.validate_receipt(stored, root=ROOT)
    assert stored["counts"]["reserved_confirmation_target_accesses"] == 0


def test_resealed_false_pass_is_rejected_if_experiment_has_run() -> None:
    config = item7.load_config(ROOT)
    path = ROOT / config["output"]
    if not path.exists():
        pytest.skip("experiment not run")
    stored = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    stored["decision"] = (
        "PASS_ITEM7_BARYONIC_COMPOSITION_EXPLORATION_REQUIRES_CONFIRMATION_AUTHORIZATION"
    )
    with pytest.raises(item7.GravityItem7BaryonicCompositionError):
        item7.validate_receipt(item7._seal(stored), root=ROOT)
