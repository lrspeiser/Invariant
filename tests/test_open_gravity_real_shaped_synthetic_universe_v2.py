from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.open_gravity_real_shaped_synthetic_universe_v2 import (
    build_catalogue,
    build_foundation_receipt,
    load_config,
    validate_config,
)
from sigma_theory_compiler.sigma_core import SchemaViolation


def test_eight_source_anchors_and_domain_populations_are_bound() -> None:
    config = load_config()
    receipt = build_foundation_receipt(config)
    assert receipt["domain_count"] == 8
    assert receipt["source_anchor_count"] == 8
    assert receipt["data_element_count"] == 62
    assert receipt["formula_replays_executed"] == 0
    assert receipt["scientific_response_rows_opened"] == 0
    assert not receipt["generators_implemented"]


def test_every_required_feature_is_visible_and_every_response_and_truth_is_hidden() -> None:
    config = load_config()
    catalogue = build_catalogue(config)
    for population in config["domains"]:
        visible = catalogue.visible_features(population["experiment_id"])
        assert set(population["required_features"]) <= visible
        assert set(population["response_features"]).isdisjoint(visible)
        assert "truth.scalar.injection-id" not in visible
        assert "truth.scalar.parameter-vector" not in visible


def test_anchor_and_feature_inventory_mutations_fail_closed() -> None:
    config = load_config()
    forged_anchor = copy.deepcopy(config)
    forged_anchor["source_anchors"][0]["sha256"] = "0" * 64
    with pytest.raises(SchemaViolation, match="source anchor changed"):
        validate_config(forged_anchor, verify_anchors=True)

    forged_feature = copy.deepcopy(config)
    forged_feature["domains"][0]["required_features"][0] = "source.scalar.unregistered"
    forged_feature["domains"][0]["required_features"].sort()
    with pytest.raises(SchemaViolation, match="feature metadata mismatch"):
        validate_config(forged_feature, verify_anchors=False)
