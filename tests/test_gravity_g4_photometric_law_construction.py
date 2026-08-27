"""Controls for the surface-brightness G4 construction repair."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g4_photometric_law_construction import (
    FEATURE_IDS,
    OUTPUT_PATH,
    GravityG4PhotometricError,
    build_receipt,
    formula_terms,
    load_config,
    prepare_photometric_packets,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def test_photometric_contract_retains_original_g4_gate() -> None:
    config = _config()
    assert config["predecessor_binding"]["required_decision"] == (
        "BLOCK_G4_EXPLORATION_CONSTRUCTION"
    )
    assert config["surface_brightness_binding"]["required_confirmation_rows"] == 0
    assert config["formula_family"]["universal_correction_constants"] == 2
    assert config["formula_family"]["per_galaxy_gravitational_constants"] == 0
    assert config["admission"]["nfw_ceiling_slack_chi_square_per_point"] == "2.0"
    assert config["admission"]["confirmation_evaluator_accesses_allowed"] == 0


def test_photometric_term_grammar_is_complete() -> None:
    terms = formula_terms()
    assert len(terms) == 90
    assert len({term_id for term_id, _factors in terms}) == 90
    assert {factor for _term_id, factors in terms for factor in factors} == set(FEATURE_IDS)


def test_photometric_features_are_target_blind() -> None:
    packet = prepare_photometric_packets(ROOT)[0]
    feature = packet["features"]["sb_log_slope"].copy()
    poisoned = dict(packet)
    poisoned["arrays"] = dict(packet["arrays"])
    poisoned["arrays"]["vobs"] = np.full_like(packet["arrays"]["vobs"], 1e99)
    poisoned["arrays"]["sigma"] = np.full_like(packet["arrays"]["sigma"], 1e-99)
    np.testing.assert_array_equal(feature, poisoned["features"]["sb_log_slope"])


def test_partial_photometric_grammar_cannot_authorize_confirmation() -> None:
    receipt = build_receipt(ROOT, term_limit=1)
    assert receipt["decision"] == "BLOCK_G4_PHOTOMETRIC_CONSTRUCTION"
    assert receipt["gate_checks"]["compact_photometric_family_fully_searched"] is False
    assert receipt["claims"]["confirmation_authorized"] is False
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0


def test_checked_photometric_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full photometric G4 construction has not completed")
    validate_receipt(json.loads(path.read_text(encoding="utf-8")), root=ROOT)


def test_checked_photometric_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full photometric G4 construction has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["historical_novelty_established"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG4PhotometricError, match="overstates novelty"):
        validate_receipt(tampered, root=ROOT)
