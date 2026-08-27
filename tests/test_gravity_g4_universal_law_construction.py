"""Controls for compact G4 universal-law construction."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g3_meta_law import prepare_packets
from sigma_theory_compiler.gravity_g3_meta_law_v2 import load_config as load_g3_config
from sigma_theory_compiler.gravity_g4_universal_law_construction import (
    FEATURE_IDS,
    OUTPUT_PATH,
    GravityG4ConstructionError,
    build_receipt,
    formula_terms,
    load_config,
    term_values,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def test_g4_contract_is_compact_universal_and_confirmation_locked() -> None:
    config = _config()
    assert config["predecessor_binding"]["required_decision"] == (
        "PASS_G3_FIXED_SHRINKAGE_META_LAW"
    )
    assert config["formula_family"]["universal_correction_constants"] == 2
    assert config["formula_family"]["per_galaxy_gravitational_constants"] == 0
    assert config["formula_family"]["term_count"] == 27
    assert config["admission"]["confirmation_evaluator_accesses_allowed"] == 0
    assert config["diagnostic_disclosure"]["result_is_independent_confirmation"] is False


def test_term_grammar_is_complete_and_excludes_shortcuts() -> None:
    terms = formula_terms()
    assert len(terms) == 27
    assert len({term_id for term_id, _factors in terms}) == 27
    assert {factor for _term_id, factors in terms for factor in factors} == set(FEATURE_IDS)
    forbidden = {"mass_proxy_fraction", "gas_to_disk", "log_distance_mpc"}
    assert not forbidden.intersection(
        factor for _term_id, factors in terms for factor in factors
    )


def test_formula_term_does_not_read_velocity_targets() -> None:
    packet = prepare_packets(ROOT, load_g3_config(ROOT))[0]
    expected = term_values(packet, ("gas_fraction", "baryon_log_slope"))
    poisoned = dict(packet)
    poisoned["arrays"] = dict(packet["arrays"])
    poisoned["arrays"]["vobs"] = np.full_like(packet["arrays"]["vobs"], 1e99)
    poisoned["arrays"]["sigma"] = np.full_like(packet["arrays"]["sigma"], 1e-99)
    np.testing.assert_array_equal(
        expected,
        term_values(poisoned, ("gas_fraction", "baryon_log_slope")),
    )


def test_partial_grammar_cannot_authorize_confirmation() -> None:
    receipt = build_receipt(ROOT, term_limit=1)
    assert receipt["decision"] == "BLOCK_G4_EXPLORATION_CONSTRUCTION"
    assert receipt["gate_checks"]["compact_family_fully_searched"] is False
    assert receipt["claims"]["confirmation_authorized"] is False
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0


def test_checked_g4_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G4 construction has not completed")
    validate_receipt(json.loads(path.read_text(encoding="utf-8")), root=ROOT)


def test_checked_g4_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G4 construction has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["historical_novelty_established"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG4ConstructionError, match="overstates novelty"):
        validate_receipt(tampered, root=ROOT)
