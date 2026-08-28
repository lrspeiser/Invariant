"""Controls for the ten-lane G4 first-principles mechanism search."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g4_first_principles_mechanism_search import (
    LANE_IDS,
    OUTPUT_PATH,
    GravityG4FirstPrinciplesError,
    build_receipt,
    load_config,
    materialize_mechanisms,
    mechanism_specs,
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


def test_mechanism_contract_freezes_nested_holdouts_and_downstream_locks() -> None:
    config = _config()
    assert config["nested_whole_galaxy_evaluation"]["outer_folds"] == 5
    assert config["candidate_accounting"]["total_candidate_structures"] == 281
    assert config["candidate_accounting"]["declared_scoring_point_evaluations"] == 49_680_800
    assert config["origin_policy"]["historical_novelty_claimed"] is False
    assert config["population"]["confirmation_evaluator_accesses_allowed"] == 0
    assert config["population"]["cluster_evaluator_accesses_allowed"] == 0
    assert config["population"]["lensing_evaluator_accesses_allowed"] == 0


def test_all_ten_lanes_are_complete_typed_and_origin_labeled() -> None:
    specs = mechanism_specs()
    counts = Counter(row["lane"] for row in specs)
    assert tuple(counts) == LANE_IDS
    assert tuple(counts.values()) == (12, 48, 48, 24, 48, 24, 16, 24, 13, 24)
    assert len(specs) == 281
    assert len({row["candidate_id"] for row in specs}) == 281
    assert sum(row["role"] == "mechanism" for row in specs) == 280
    assert sum(row["role"] == "known_positive_control" for row in specs) == 1
    assert all(row["equation_ir"]["dimension_output"] == "velocity_squared" for row in specs)
    assert all("origin_label" in row for row in specs)


def test_every_mechanism_lane_is_target_blind() -> None:
    packet = _packets()[0]
    poisoned = copy.deepcopy(packet)
    poisoned["arrays"]["vobs"] = np.full_like(packet["arrays"]["vobs"], 1e99)
    poisoned["arrays"]["sigma"] = np.full_like(packet["arrays"]["sigma"], 1e-99)
    clean = materialize_mechanisms([packet])
    tainted = materialize_mechanisms([poisoned])
    for left, right in zip(clean, tainted, strict=True):
        assert left["candidate_id"] == right["candidate_id"]
        np.testing.assert_array_equal(left["component_v2"], right["component_v2"])


def test_partial_mechanism_search_cannot_authorize_confirmation() -> None:
    receipt = build_receipt(ROOT, candidate_limit=3)
    assert receipt["decision"] == "BLOCK_G4_FIRST_PRINCIPLES_MECHANISM_SEARCH"
    assert receipt["gate_checks"]["complete_ten_lane_grammar_searched"] is False
    assert receipt["gate_checks"]["complete_first_principles_obligations"] is False
    assert receipt["claims"]["confirmation_authorized"] is False
    assert receipt["counts"]["candidate_structures"] == 3
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0
    assert receipt["counts"]["cross_scale_cluster_evaluator_accesses"] == 0
    assert receipt["counts"]["cross_scale_lensing_evaluator_accesses"] == 0


def test_checked_mechanism_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full first-principles mechanism search has not completed")
    validate_receipt(json.loads(path.read_text(encoding="utf-8")), root=ROOT)


def test_checked_mechanism_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full first-principles mechanism search has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["covariant_first_principles_theory_derived"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG4FirstPrinciplesError, match="overstates derivation"):
        validate_receipt(tampered, root=ROOT)
