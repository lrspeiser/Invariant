"""Controls for the G4 conditional formula-of-formulas generator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g4_conditional_formula_generator import (
    CONDITION_IDS,
    OUTPUT_PATH,
    GravityG4ConditionalGeneratorError,
    build_receipt,
    concept_operators,
    galaxy_conditions,
    load_config,
    validate_receipt,
)
from sigma_theory_compiler.gravity_g4_nonlocal_profile_law_construction import (
    prepare_nonlocal_packets,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def _packets() -> Any:
    if "packets" not in _CACHE:
        _CACHE["packets"] = prepare_nonlocal_packets(ROOT)
    return _CACHE["packets"]


def test_conditional_contract_binds_atlas_and_keeps_confirmation_locked() -> None:
    config = _config()
    assert config["g2_binding"]["required_structural_classes"] == 8609
    assert config["atlas_lane"]["stage_a_class_condition_pairs"] == 60263
    assert config["candidate_accounting"]["total_declared_cells"] == 261348
    assert config["admission"]["confirmation_evaluator_accesses_allowed"] == 0
    assert config["baryonic_focusing_lane"]["historical_novelty_claimed"] is False
    assert config["secular_synchronization_lane"]["historical_novelty_claimed"] is False


def test_conditions_are_bounded_and_target_blind() -> None:
    packet = _packets()[0]
    expected = galaxy_conditions(packet)
    assert tuple(expected) == CONDITION_IDS
    assert all(np.isfinite(value) and -1.0 <= value <= 1.0 for value in expected.values())

    poisoned = copy.deepcopy(packet)
    poisoned["arrays"]["vobs"] = np.full_like(packet["arrays"]["vobs"], 1e99)
    poisoned["arrays"]["sigma"] = np.full_like(packet["arrays"]["sigma"], 1e-99)
    assert galaxy_conditions(poisoned) == expected


def test_concept_grammar_is_complete_and_target_blind() -> None:
    packets = _packets()
    operators = concept_operators(packets)
    assert len(operators) == 80
    assert len({row["operator_id"] for row in operators}) == 80
    assert sum(row["family"] == "baryonic_focusing" for row in operators) == 48
    assert sum(row["family"] == "secular_speed_synchronization" for row in operators) == 32

    poisoned = copy.deepcopy(packets[0])
    poisoned["arrays"]["vobs"] = np.full_like(poisoned["arrays"]["vobs"], 1e99)
    poisoned["arrays"]["sigma"] = np.full_like(poisoned["arrays"]["sigma"], 1e-99)
    clean = concept_operators([packets[0]], operator_limit=2)
    tainted = concept_operators([poisoned], operator_limit=2)
    for left, right in zip(clean, tainted, strict=True):
        np.testing.assert_array_equal(left["component_v2"], right["component_v2"])


def test_partial_generator_cannot_authorize_confirmation() -> None:
    receipt = build_receipt(ROOT, class_limit=2, operator_limit=2)
    assert receipt["decision"] == "BLOCK_G4_CONDITIONAL_GENERATOR"
    assert receipt["gate_checks"]["complete_conditional_generator_grammar_searched"] is False
    assert receipt["claims"]["confirmation_authorized"] is False
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0
    assert receipt["counts"]["atlas_classes_audited"] == 2
    assert receipt["counts"]["atlas_stage_b_parent_pairs"] == 8
    assert receipt["counts"]["atlas_stage_b_cells"] == 64
    assert receipt["counts"]["concept_operators"] == 2


def test_checked_conditional_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full conditional generator has not completed")
    validate_receipt(json.loads(path.read_text(encoding="utf-8")), root=ROOT)


def test_checked_conditional_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full conditional generator has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["resonance_dynamics_derived"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG4ConditionalGeneratorError, match="overstates resonance"):
        validate_receipt(tampered, root=ROOT)
