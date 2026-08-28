from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item8_field_gradients_curvature as item8

ROOT = Path(__file__).resolve().parents[1]


def test_config_binds_item7_and_forbids_dynamical_shortcuts() -> None:
    config = item8.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 8
    assert config["predecessor"]["required_decision"].endswith("ADVANCE_ITEM8")
    assert config["prefreeze_audit"]["published_group_dispersion_values_read"] == 0
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["authorization"]["reserved_confirmation_group_dispersion_allowed"] is False
    assert config["authorization"]["member_redshift_allowed"] is False
    assert config["authorization"]["virial_or_halo_mass_allowed_as_predictor"] is False
    assert config["authorization"]["virial_radius_allowed_as_predictor"] is False
    assert config["authorization"]["lensing_mass_allowed_as_predictor"] is False


def test_sample_is_response_blind_and_sealed() -> None:
    config = item8.load_config(ROOT)
    manifest = item8.build_sample_manifest(ROOT)
    item8.validate_sample_manifest(manifest, config)
    assert manifest["counts"] == {
        "catalog_groups": 813,
        "catalog_members": 4869,
        "eligible": 131,
        "exploration": 98,
        "reserved_confirmation": 33,
    }
    folds = [row["outer_fold"] for row in manifest["objects"] if row["role"] == "exploration"]
    assert sorted(folds.count(index) for index in range(5)) == [19, 19, 20, 20, 20]
    assert manifest["selection"]["selection_used_group_dispersion"] is False
    assert manifest["selection"]["selection_used_member_redshift"] is False
    assert manifest["prefreeze_boundary"]["reserved_confirmation_dispersions_blinded"]


def _synthetic_features(rotation_deg: float = 0.0) -> dict[str, float]:
    offsets = np.asarray(
        [
            [-0.30, -0.12],
            [-0.18, 0.21],
            [-0.05, -0.27],
            [0.09, 0.31],
            [0.19, -0.18],
            [0.28, 0.07],
            [0.37, 0.24],
            [-0.34, 0.33],
        ]
    )
    angle = np.deg2rad(rotation_deg)
    matrix = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = offsets @ matrix.T
    return item8.measure_field_features(
        ra_deg=(10.0 + rotated[:, 0]).tolist(),
        dec_deg=rotated[:, 1].tolist(),
        k_magnitude=[9.8, 10.1, 10.4, 9.9, 10.8, 11.1, 10.2, 11.4],
        mean_velocity_km_s=7000.0,
        config=item8.load_config(ROOT),
    )


def test_feature_builder_is_response_blind_finite_and_rotation_invariant() -> None:
    parameters = " ".join(inspect.signature(item8.measure_field_features).parameters).lower()
    assert "dispersion" not in parameters
    assert "member_redshift" not in parameters
    original = _synthetic_features()
    rotated = _synthetic_features(41.0)
    assert set(original) == set(item8.FEATURE_NAMES)
    assert all(np.isfinite(value) for value in original.values())
    for field in item8.FEATURE_NAMES:
        assert rotated[field] == pytest.approx(original[field], rel=2e-4, abs=2e-4)


def test_creativity_boundary_keeps_rewrites_and_shape_as_controls() -> None:
    models = {row["id"]: row for row in item8.load_config(ROOT)["model_families"]}
    assert models["virial_fixed"]["qualifying"] is False
    assert models["mass_size_nuisance"]["qualifying"] is False
    assert models["mass_size_shape_controls"]["qualifying"] is False
    assert models["normalized_center_tidal_invariants"]["qualifying"] is True
    assert models["third_derivative_alignment"]["qualifying"] is True
    assert models["radial_field_curvature"]["qualifying"] is True


def test_predictor_queries_exclude_response_and_virial_shortcuts() -> None:
    config = item8.load_config(ROOT)
    source = config["sources"]
    urls = (
        item8._query_url(
            source["group_table"], columns=source["allowed_group_predictor_columns"]
        ),
        item8._query_url(
            source["member_table"], columns=source["allowed_member_predictor_columns"]
        ),
    )
    for url in urls:
        lowered = url.lower()
        for forbidden in ("sigma", "rvir", "mvir"):
            assert forbidden not in lowered
    assert "-out=z" not in urls[1]


def test_synthetic_source_parsers() -> None:
    groups = item8.parse_group_predictors(b"11\t8\t7000.0\n")
    assert groups[11] == {
        "group": 11,
        "members": 8,
        "mean_velocity_km_s": 7000.0,
    }
    members = item8.parse_member_predictors(
        b"11\t1\t10.2\t-0.3\t9.8\n11\t2\t10.3\t-0.2\t10.1\n"
    )
    assert len(members[11]) == 2
    assert members[11][0]["k_magnitude"] == 9.8
    assert item8.parse_group_response(b"11\t210.0\n", expected_group=11) == 210.0


def test_response_acquisition_is_impossible_before_freeze_binding() -> None:
    if item8.SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(item8.GravityItem8FieldCurvatureError, match="not bound"):
            item8.acquire_responses(ROOT)


def test_stored_target_blind_artifacts_replay_after_they_are_written() -> None:
    config = item8.load_config(ROOT)
    sample_path = ROOT / config["sample_manifest_output"]
    if sample_path.exists():
        assert json.loads(sample_path.read_text(encoding="utf-8")) == item8.build_sample_manifest(
            ROOT
        )
    source_path = ROOT / config["predictor_source_output"]
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        item8.validate_predictor_source(source, config)
        assert source["boundary"]["published_group_dispersion_values_acquired"] == 0
        assert source["boundary"]["member_redshift_values_acquired"] == 0
    feature_path = ROOT / config["feature_output"]
    if feature_path.exists():
        header = feature_path.read_text(encoding="utf-8").splitlines()[0].lower()
        assert "sigma" not in header
        assert "member_redshift" not in header


def test_response_manifest_keeps_confirmation_closed_if_acquired() -> None:
    config = item8.load_config(ROOT)
    path = ROOT / config["response_source_output"]
    if not path.exists():
        pytest.skip("exploration responses not acquired")
    source = json.loads(path.read_text(encoding="utf-8"))
    item8.validate_response_source(source, config)
    assert source["boundary"]["reserved_confirmation_dispersion_queries"] == 0
    assert source["boundary"]["member_redshift_values_acquired"] == 0
    assert source["boundary"]["virial_or_halo_mass_values_acquired"] == 0


def test_receipt_replays_if_experiment_has_run() -> None:
    config = item8.load_config(ROOT)
    path = ROOT / config["output"]
    if not path.exists():
        pytest.skip("experiment not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert item8.build_receipt(ROOT) == stored
    item8.validate_receipt(stored, root=ROOT)
    assert stored["counts"]["reserved_confirmation_target_accesses"] == 0
    assert stored["counts"]["post_response_formula_generation_allowed"] == 0
