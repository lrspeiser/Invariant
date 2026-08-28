from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    gravity_item3_smooth_density_profiles as item3,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_binds_roadmap_and_failed_attempt() -> None:
    config = item3.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 3
    assert config["predecessor"]["required_decision"] == (
        "INCONCLUSIVE_ITEM3_SURFACE_VOLUME_DENSITY_QUALITY_GATE"
    )
    assert config["authorization"]["reserved_xcop_confirmation_profile_accesses_allowed"] == 0
    assert config["cluster_lane"]["reserved_profile_accesses_allowed"] == 0


@pytest.mark.parametrize("support_dimension", [2, 3])
def test_effective_pair_has_frozen_identity(support_dimension: int) -> None:
    radius = np.geomspace(0.5, 20.0, 31)
    density = 10.0 * np.exp(-radius / 4.0)
    pair = item3.effective_density_pair(
        radius,
        density,
        support_dimension=support_dimension,
        gravity_constant=1.0,
        transition_acceleration=2.0,
    )
    expected_contrast = np.log10(3.0 * pair["scale_length"] / (2.0 * radius))
    np.testing.assert_allclose(pair["c"], expected_contrast, rtol=1.0e-12, atol=1.0e-12)
    assert np.all(pair["u_surface"] > 0)
    assert np.all(pair["u_volume"] > 0)


def test_feature_builder_cannot_accept_a_target() -> None:
    radius = np.geomspace(1.0, 10.0, 12)
    density = 20.0 * radius**-1.2
    pair = item3.effective_density_pair(
        radius,
        density,
        support_dimension=2,
        gravity_constant=1.0,
        transition_acceleration=1.0,
    )
    features = item3.radial_feature_basis(
        gbar=radius**-1,
        density_pair=pair,
        transition_acceleration=1.0,
        population_proxy=0.0,
    )
    assert "target" not in features
    assert "response" not in features
    assert set(features) == {
        "a",
        "a2",
        "a3",
        "s",
        "v",
        "m",
        "c",
        "m_x_c",
        "a_x_m",
        "a_x_c",
        "transition_product",
        "transition_balance",
        "population_proxy",
    }


def test_salted_cluster_split_and_confirmation_budget() -> None:
    manifest = item3.build_sample_manifest(ROOT)
    assert manifest["counts"] == {
        "galaxy_development": 11,
        "cluster_exploration": 8,
        "cluster_reserved_confirmation": 4,
    }
    assert manifest["confirmation_access_budget"] == 0
    assert set(manifest["cluster_exploration"]).isdisjoint(
        manifest["cluster_reserved_confirmation"]
    )
    content = dict(manifest)
    content.pop("content_sha256")
    expected_hash = item3.canonical_sha256(content)
    assert manifest["content_sha256"] == expected_hash


def test_committed_manifest_matches_builder() -> None:
    path = ROOT / item3.SAMPLE_MANIFEST_PATH
    if not path.exists():
        pytest.skip("manifest is generated immediately before the freeze commit")
    assert json.loads(path.read_text(encoding="utf-8")) == item3.build_sample_manifest(ROOT)
