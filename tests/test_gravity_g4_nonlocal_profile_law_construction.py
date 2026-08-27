"""Controls for the real-data G4 nonlocal radial-profile construction."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g4_nonlocal_profile_law_construction import (
    OUTPUT_PATH,
    GravityG4NonlocalError,
    build_receipt,
    feature_specs,
    load_config,
    materialize_nonlocal_features,
    prepare_nonlocal_packets,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def test_nonlocal_contract_retains_g4_gate_and_confirmation_lock() -> None:
    config = _config()
    assert config["predecessor_binding"]["required_decision"] == (
        "BLOCK_G4_PHOTOMETRIC_CONSTRUCTION"
    )
    assert config["surface_brightness_binding"]["required_confirmation_rows"] == 0
    assert config["formula_family"]["feature_count"] == 360
    assert config["formula_family"]["cascade"]["formula_structures"] == 636
    assert config["formula_family"]["per_galaxy_gravitational_constants"] == 0
    assert config["admission"]["confirmation_evaluator_accesses_allowed"] == 0
    assert config["origin_assessment"]["historical_novelty_claimed"] is False


def test_nonlocal_feature_grammar_is_complete_and_unique() -> None:
    specs = feature_specs()
    assert len(specs) == 360
    assert len({row["feature_id"] for row in specs}) == 360
    assert {row["representation"] for row in specs} == {
        "weighted_mean",
        "mean_minus_local",
    }


def test_nonlocal_features_are_target_blind_and_distance_scale_invariant() -> None:
    packet = prepare_nonlocal_packets(ROOT)[0]
    feature_id = feature_specs()[17]["feature_id"]
    feature = packet["nonlocal_features"][feature_id].copy()
    poisoned = copy.deepcopy(packet)
    poisoned["arrays"]["vobs"] = np.full_like(packet["arrays"]["vobs"], 1e99)
    poisoned["arrays"]["sigma"] = np.full_like(packet["arrays"]["sigma"], 1e-99)
    np.testing.assert_array_equal(
        feature, materialize_nonlocal_features(poisoned)[feature_id]
    )

    scaled = copy.deepcopy(packet)
    scaled["arrays"]["radius"] = packet["arrays"]["radius"] * 7.0
    np.testing.assert_allclose(
        feature,
        materialize_nonlocal_features(scaled)[feature_id],
        rtol=0.0,
        atol=2e-14,
    )


def test_partial_nonlocal_cascade_cannot_authorize_confirmation() -> None:
    receipt = build_receipt(ROOT, feature_limit=3)
    assert receipt["decision"] == "BLOCK_G4_NONLOCAL_PROFILE_CONSTRUCTION"
    assert receipt["gate_checks"]["full_nonlocal_cascade_searched"] is False
    assert receipt["claims"]["confirmation_authorized"] is False
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0
    assert receipt["counts"]["formula_structures"] == 6


def test_checked_nonlocal_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G4 nonlocal construction has not completed")
    validate_receipt(json.loads(path.read_text(encoding="utf-8")), root=ROOT)


def test_checked_nonlocal_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G4 nonlocal construction has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["historical_novelty_established"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG4NonlocalError, match="overstates novelty"):
        validate_receipt(tampered, root=ROOT)
