from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item9_interior_exterior as item9

ROOT = Path(__file__).resolve().parents[1]


def test_config_binds_item8_and_forbids_response_shortcuts() -> None:
    config = item9.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 9
    assert config["predecessor"]["required_decision"].endswith("ADVANCE_ITEM9")
    assert config["prefreeze_audit"]["published_rotation_profile_rows_read"] == 0
    assert config["prefreeze_audit"]["profile_archive_bytes_downloaded"] == 0
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["authorization"]["reserved_confirmation_rotation_profiles_allowed"] is False
    assert config["authorization"]["model_fits_table_allowed"] is False
    assert config["authorization"]["structural_parameters_table_allowed"] is False
    assert config["authorization"]["dynamical_or_dark_mass_allowed"] is False
    assert config["authorization"]["lensing_mass_allowed"] is False
    assert config["authorization"]["post_response_formula_generation_allowed"] is False


def test_stored_metadata_and_sample_are_response_blind() -> None:
    config = item9.load_config(ROOT)
    metadata_path = ROOT / config["metadata_source_output"]
    sample_path = ROOT / config["sample_manifest_output"]
    if not metadata_path.exists() or not sample_path.exists():
        pytest.skip("target-blind Item 9 artifacts not written")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    item9.validate_metadata_source(metadata, config)
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    item9.validate_sample_manifest(sample, config)
    assert sample["counts"] == {
        "eligible": 1292,
        "exploration": 969,
        "reserved_confirmation": 323,
        "rotation_profile_rows_read": 0,
        "profile_archive_bytes_downloaded": 0,
    }
    assert len(sample["cells"]) == 18
    folds = [
        int(row["outer_fold"])
        for row in sample["objects"]
        if row["role"] == "exploration"
    ]
    assert sorted(folds.count(index) for index in range(5)) == [193, 194, 194, 194, 194]
    assert not any("SPARC" in row["rc_survey"].upper() for row in sample["objects"])


def test_candidate_manifest_counts_labels_and_prior_cell() -> None:
    config = item9.load_config(ROOT)
    manifest = item9.build_candidate_manifest(ROOT)
    item9.validate_candidate_manifest(manifest, config)
    assert manifest["counts"]["operators"] == 72
    assert manifest["counts"]["candidate_formula_cells"] == 12600
    assert manifest["counts"]["declared_equivalence_classes"] == 11160
    assert manifest["counts"]["origin_status_cells"] == {
        "COMBINATION": 8400,
        "KNOWN_FAMILY_COMBINATION": 1400,
        "UNRESOLVED": 2800,
    }
    prior = [row for row in manifest["cells"] if row["exact_prior_focusing_cell"]]
    assert len(prior) == 1
    assert prior[0]["candidate_id"].endswith("surface_density:am2:b4")
    assert prior[0]["authoritative_origin_status"] == "COMBINATION"
    assert not any(row["historical_novelty_claimed"] for row in manifest["cells"])
    stored_path = ROOT / config["candidate_manifest_output"]
    if stored_path.exists():
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        assert stored == manifest
        thresholds = {
            None if row["threshold"] is None else float(row["threshold"])
            for row in stored["operators"]
        }
        assert {None, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}.issubset(thresholds)


def _synthetic_photometry() -> list[dict[str, float]]:
    rows = []
    for radius in range(1, 31):
        rows.append(
            {
                "R": float(radius),
                "SB": 20.0 + 0.12 * radius,
                "SB_e": 0.05,
                "totmag": 16.0 - 2.5 * math.log10(radius),
                "totmag_e": 0.03,
                "ellip": 0.5,
            }
        )
    return rows


def test_feature_builder_is_response_blind_and_finite() -> None:
    signature = " ".join(inspect.signature(item9.measure_point_features).parameters).lower()
    assert "velocity" not in signature
    assert "response" not in signature
    config = item9.load_config(ROOT)
    predictor = item9.measure_photometric_predictors(
        photometry_rows=_synthetic_photometry(),
        distance_mpc=50.0,
        extinction_r_magnitude=0.1,
        config=config,
    )
    assert 30.0 < predictor["inclination_degrees"] < 80.0
    features = item9.measure_point_features(
        predictor=predictor,
        rotation_radius_arcsec=np.concatenate(([0.0], np.linspace(2.0, 28.0, 20))),
        distance_mpc=50.0,
        config=config,
    )
    assert len(features) == 21
    assert features[0]["within_photometry"] is False
    assert all(feature["within_photometry"] for feature in features[1:])
    for feature in features:
        assert set(item9.POINT_FEATURE_FIELDS).issubset(feature)
        assert all(np.isfinite(float(feature[field])) for field in item9.POINT_FEATURE_FIELDS)


def test_profile_parsers_and_kernel_directionality() -> None:
    photometry = (
        b"arcsec,mag,mag,mag,mag,unitless\n"
        b"R,SB,SB_e,totmag,totmag_e,ellip\n"
        b"1,20,0.1,15,0.1,0.4\n2,21,0.1,14.5,0.1,0.5\n"
    )
    rotation = (
        b"arcsec,km/s,km/s\nR,V,V_e\n"
        b"-2,-80,4\n2,82,5\n"
    )
    assert len(item9.parse_photometry_profile(photometry)) == 2
    assert item9.parse_rotation_profile(rotation)[0]["V"] == -80.0
    log_radius = np.log(np.asarray([1.0, 2.0, 4.0]))
    interior = item9._kernel_matrix(log_radius, "interior", 0.5)
    exterior = item9._kernel_matrix(log_radius, "exterior", 0.5)
    assert np.allclose(np.sum(interior, axis=1), 1.0)
    assert np.allclose(np.sum(exterior, axis=1), 1.0)
    assert interior[0, 1] == 0.0
    assert exterior[-1, -2] == 0.0


def test_profile_archive_acquisition_is_impossible_before_freeze_binding() -> None:
    if item9.SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(item9.GravityItem9InteriorExteriorError, match="not bound"):
            item9.acquire_profile_archive(ROOT)


def test_candidate_scoring_and_local_control_on_synthetic_galaxies() -> None:
    config = item9.load_config(ROOT)
    candidates = item9.build_candidate_manifest(ROOT)
    names = [f"G{index:02d}" for index in range(10)]
    galaxy_index = np.repeat(np.arange(10), 3)
    point_counts = np.full(10, 3, dtype=np.int64)
    starts = np.arange(0, 30, 3, dtype=np.int64)
    radius = np.tile(np.asarray([0.5, 1.0, 2.0]), 10)
    galaxy_mass = np.repeat(np.linspace(9.0, 11.0, 10), 3)
    arrays = {
        field: np.zeros(30, dtype=np.float64) for field in item9.POINT_FEATURE_FIELDS
    }
    arrays.update(
        {
            "radius_kpc": radius,
            "log10_gbar": np.log10(2000.0 / radius),
            "log10_radius_over_r50": np.log10(radius),
            "log10_surface_density": np.log10(100.0 / radius),
            "enclosed_mass_fraction": np.tile(np.asarray([0.25, 0.6, 0.9]), 10),
            "log10_total_mass": galaxy_mass,
            "inclination_sine": np.full(30, 0.8),
            "log10_distance": np.repeat(np.linspace(1.0, 2.0, 10), 3),
            "rar_speed_km_s": 80.0 + 8.0 * galaxy_mass + 2.0 * radius,
        }
    )
    for name in item9.CONDITION_FIELD.values():
        arrays[name] = np.repeat(np.linspace(-0.8, 0.8, 10), 3)
    y = np.log10(arrays["rar_speed_km_s"] * (1.0 + 0.01 * np.sin(radius)))
    features = [
        {
            "galaxy": names[int(index)],
            "point_index": str(point),
            "primary_rc_survey": "synthetic",
            "distance_bin": "0",
            "mass_stratum": "low_mass" if index < 5 else "high_mass",
        }
        for index in range(10)
        for point in range(3)
    ]
    data = {
        "candidate_manifest": candidates,
        "names": names,
        "galaxy_index": galaxy_index,
        "point_counts": point_counts,
        "starts": starts,
        "folds": np.arange(10) % 5,
        "y": y,
        "feature_arrays": arrays,
        "features": features,
    }
    components = np.full((72, 30), 1e-3, dtype=np.float64)
    losses, invalid, compute = item9._candidate_galaxy_losses(data, components, config)
    assert losses.shape == (12600, 10)
    assert invalid.shape == (12600,)
    assert compute["candidate_point_evaluations"] == 378000
    assert compute["backend"] in {"cpu_numpy", "gpu_cupy"}
    local, folds = item9._local_control_oof(data, config)
    assert local.shape == (30,)
    assert np.all(np.isfinite(local))
    assert len(folds) == 5


def test_stored_profile_receipt_keeps_confirmation_entries_unopened() -> None:
    config = item9.load_config(ROOT)
    path = ROOT / config["profile_source_output"]
    if not path.exists():
        pytest.skip("profile archive not opened")
    source = json.loads(path.read_text(encoding="utf-8"))
    item9.validate_profile_source(source, config)
    assert source["counts"]["reserved_confirmation_rotation_entries_opened"] == 0
    confirmation = [row for row in source["records"] if row["role"] == "reserved_confirmation"]
    assert len(confirmation) == 323
    assert not any(row["response_entry_opened"] for row in confirmation)


def test_receipt_replays_if_experiment_has_run() -> None:
    config = item9.load_config(ROOT)
    path = ROOT / config["output"]
    if not path.exists():
        pytest.skip("Item 9 experiment not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    item9.validate_receipt(stored, root=ROOT)
    assert stored["counts"]["reserved_confirmation_rotation_entries_opened"] == 0
    assert stored["counts"]["post_response_formula_cells"] == 0
    assert stored["counts"]["candidate_formula_cells"] == 12600


def test_full_receipt_check_ignores_runtime_only_elapsed_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    config = item9.load_config(ROOT)
    path = ROOT / config["output"]
    if not path.exists():
        pytest.skip("Item 9 experiment not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = json.loads(json.dumps(stored))
    rebuilt["compute"]["elapsed_seconds"] = "9.999000000000e+00"
    rebuilt.pop("content_sha256")
    rebuilt["content_sha256"] = item9.canonical_sha256(rebuilt)
    monkeypatch.setattr(item9, "build_receipt", lambda root: rebuilt)
    item9.check_receipt(ROOT)

    rebuilt["primary"]["qualifying_selector"]["metrics"]["mse"] = "9.999000000000e+00"
    rebuilt.pop("content_sha256")
    rebuilt["content_sha256"] = item9.canonical_sha256(rebuilt)
    with pytest.raises(item9.GravityItem9InteriorExteriorError, match="receipt drifted"):
        item9.check_receipt(ROOT)
