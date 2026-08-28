from __future__ import annotations

import copy
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.gravity_item3_surface_volume_density as density
import sigma_theory_compiler.gravity_item3_surface_volume_density_experiment as experiment
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / density.SAMPLE_MANIFEST_PATH


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def test_item3_is_frozen_before_features_and_fresh_responses() -> None:
    config = density.load_config(ROOT)
    assert config["status"] == "frozen_before_item3_feature_or_fresh_group_response_access"
    assert config["authorization"]["fresh_exploration_group_member_rows_allowed"] is True
    assert config["authorization"]["reserved_confirmation_group_member_rows_allowed"] is False
    assert config["fresh_group_lane"]["reserved_confirmation_target_accesses_allowed"] == 0
    assert config["authorization"]["paid_model_calls_allowed"] is False


def test_surface_volume_derivation_is_dimensionless_and_excludes_item1_rewrite() -> None:
    config = density.load_config(ROOT)
    contract = config["scientific_contract"]
    assert "u_surface/u_volume=3/D_M" in contract["exact_rewrite_excluded"]
    assert config["claim_boundaries"]["surface_volume_ratio_is_new_relative_to_item1"] is False
    source = inspect.getsource(density.surface_volume_profile_features)
    assert "observed" not in source
    assert "beta" not in source
    assert "velocity" not in source


def test_uniform_sphere_has_identical_surface_and_volume_source_transitions() -> None:
    radius = np.geomspace(0.05, 20.0, 101)
    g_dagger = 1.2e-10
    gbar = g_dagger * radius / 1.0
    features = density.surface_volume_profile_features(radius, gbar, g_dagger)
    assert features["local_mass_dimension_median"] == pytest.approx(3.0, abs=1.0e-10)
    assert features["surface_volume_log_ratio_median"] == pytest.approx(0.0, abs=1.0e-10)
    assert features["log_transition_radius_ratio"] == pytest.approx(0.0, abs=1.0e-10)
    assert features["transition_overlap_cosine"] == pytest.approx(1.0, abs=1.0e-12)


def test_inverse_radius_profile_separates_the_two_threshold_locations() -> None:
    radius = np.geomspace(0.01, 100.0, 301)
    g_dagger = 1.2e-10
    gbar = g_dagger / radius
    features = density.surface_volume_profile_features(radius, gbar, g_dagger)
    assert features["local_mass_dimension_median"] == pytest.approx(1.0, abs=1.0e-10)
    assert features["surface_volume_log_ratio_median"] == pytest.approx(
        np.log(3.0), abs=1.0e-10
    )
    assert features["log_transition_radius_ratio"] < -0.8
    assert features["transition_overlap_cosine"] < 1.0


def test_fresh_group_sample_is_balanced_disjoint_and_target_blind() -> None:
    config = density.load_config(ROOT)
    manifest = _load(SAMPLE)
    density.validate_sample_manifest(manifest, config=config)
    assert density.build_sample_manifest(ROOT) == manifest
    objects = manifest["objects"]
    assert len(objects) == 180
    assert Counter(row["role"] for row in objects) == {
        "exploration": 120,
        "reserved_confirmation": 60,
    }
    assert Counter((row["richness_bin"], row["role"]) for row in objects) == {
        (richness_bin, role): count
        for richness_bin in range(3)
        for role, count in (("exploration", 40), ("reserved_confirmation", 20))
    }
    item2 = _load(ROOT / config["fresh_group_lane"]["item2_exclusion_manifest"])
    assert {row["group"] for row in objects}.isdisjoint(
        {row["group"] for row in item2["objects"]}
    )
    assert manifest["selection_boundary"]["fresh_member_rows_opened"] == 0
    assert manifest["selection_boundary"]["fresh_member_redshifts_read"] == 0
    assert manifest["selection_boundary"]["reserved_confirmation_target_accesses"] == 0


@pytest.mark.parametrize(
    "claim",
    [
        "alternative_to_gr_established",
        "confirmation_opened",
        "item2_groups_reused",
        "member_response_seen_during_selection",
        "roadmap_item_3_complete",
    ],
)
def test_resealed_sample_overclaim_is_rejected(claim: str) -> None:
    config = density.load_config(ROOT)
    manifest = copy.deepcopy(_load(SAMPLE))
    manifest["claims"][claim] = True
    with pytest.raises(density.GravityItem3SurfaceVolumeDensityError):
        density.validate_sample_manifest(_reseal(manifest), config=config)


def test_fresh_source_acquisition_never_reuses_item2_or_confirmation() -> None:
    config = density.load_config(ROOT)
    sample = _load(SAMPLE)
    manifest = _load(ROOT / density.SOURCE_MANIFEST_PATH)
    density.validate_source_manifest(manifest, sample=sample)
    assert manifest["counts"] == {"bytes": 549153, "groups": 120, "member_rows": 3096}
    assert manifest["boundary"] == {
        "fresh_exploration_groups_acquired": 120,
        "fresh_exploration_target_accesses": 120,
        "item2_group_target_reuse": 0,
        "published_group_velocity_columns_read": 0,
        "reserved_confirmation_groups_acquired": 0,
        "reserved_confirmation_target_accesses": 0,
    }
    assert {row["group"] for row in manifest["records"]} == {
        row["group"] for row in sample["objects"] if row["role"] == "exploration"
    }
    assert config["authorization"]["reserved_confirmation_group_member_rows_allowed"] is False


def test_group_density_extractor_cannot_accept_member_redshifts() -> None:
    signature = inspect.signature(density.measure_group_density_only)
    assert tuple(signature.parameters) == (
        "ra_deg",
        "dec_deg",
        "luminosity",
        "metadata_redshift",
        "config",
    )
    source = inspect.getsource(density.measure_group_density_only)
    assert "member_redshift" not in source
    assert "velocity" not in source
    assert "sigma" not in source


def test_extraction_retains_frozen_failures_and_both_lanes() -> None:
    config = density.load_config(ROOT)
    summary = _load(ROOT / density.EXTRACTION_SUMMARY_PATH)
    assert summary["decision"] == "FAIL_ITEM3_EXPLORATION_REPRESENTATION_QUALITY"
    assert summary["counts"] == {
        "cross_scale_failures": 11,
        "cross_scale_passing": 148,
        "fresh_group_failures": 33,
        "fresh_group_passing": 87,
        "reserved_confirmation_target_accesses": 0,
    }
    assert len({(row["domain"], row["name"]) for row in summary["cross_scale_failures"]}) == 11
    assert len({row["group"] for row in summary["fresh_group_failures"]}) == 33
    assert {row["reason"] for row in summary["fresh_group_failures"]} == {
        "non-strict luminosity quantile radii"
    }
    cross_rows = experiment._load_cross_rows(ROOT, config)
    group_rows = experiment._load_group_rows(ROOT, config)
    assert Counter(row["domain"] for row in cross_rows) == {"galaxy": 137, "cluster": 11}
    assert Counter(row["richness_bin"] for row in group_rows) == {0: 18, 1: 29, 2: 40}


def test_item3_receipt_replays_and_records_negative_increment() -> None:
    stored = _load(ROOT / experiment.OUTPUT_PATH)
    assert experiment.build_receipt(ROOT) == stored
    experiment.validate_receipt(stored, root=ROOT)
    assert stored["decision"] == "INCONCLUSIVE_ITEM3_SURFACE_VOLUME_DENSITY_QUALITY_GATE"
    assert stored["counts"]["fresh_group_confirmation_target_accesses"] == 0
    assert stored["cross_scale_result"]["fold_ledger"] == [
        {
            "alpha": "1.000000000000e+00",
            "fold": fold,
            "heldout_objects": count,
            "inner_mse": inner_mse,
            "model_id": "binary_population_proxy",
            "qualifying": False,
        }
        for fold, count, inner_mse in (
            (0, 31, "8.028683534986e-02"),
            (1, 30, "7.641416948719e-02"),
            (2, 29, "7.441940005724e-02"),
            (3, 29, "6.807965650573e-02"),
            (4, 29, "7.340176341117e-02"),
        )
    ]
    baseline = stored["fresh_group_result"]["model_metrics"]["strongest_nuisance_baseline"]
    augmented = stored["fresh_group_result"]["model_metrics"][
        "surface_volume_density_augmented"
    ]
    assert float(baseline["overall"]["r2"]) > float(augmented["overall"]["r2"])
    assert float(stored["fresh_group_result"]["permutation_test"]["p_value"]) == 0.795
    assert stored["gate_checks"]["confirmation_untouched"] is True
    assert sum(stored["gate_checks"].values()) == 1


def test_resealed_false_item3_pass_is_rejected() -> None:
    stored = copy.deepcopy(_load(ROOT / experiment.OUTPUT_PATH))
    stored["decision"] = "PASS_ITEM3_SURFACE_VOLUME_DENSITY_EXPLORATION_REQUIRES_AUTHORIZATION"
    with pytest.raises(experiment.GravityItem3ExperimentError):
        experiment.validate_receipt(_reseal(stored), root=ROOT)
