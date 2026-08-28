from __future__ import annotations

import copy
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.gravity_item3_surface_volume_density as density
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
