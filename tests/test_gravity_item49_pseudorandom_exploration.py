from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    _best_behavior,
    _lane_ordinals,
    build_aggregate_result,
    build_candidate_manifest,
    build_evaluation_result,
    build_exposure_manifest,
    build_primitive_receipt,
    decode_ordinals,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_item49_frozen_space_and_equal_schedule_budget() -> None:
    config = load_config(ROOT)
    assert config["program_grammar"]["full_ordinal_space"] == 6_496_138_035_200
    assert config["primitive_bank"]["total_primitives"] == 440
    assert config["schedules"]["pseudorandom"]["sample_positions"] == 1_048_576
    assert (
        config["schedules"]["sequential_ordinal_control"]["sample_ordinals"]
        == 1_048_576
    )
    assert config["scope"]["full_grammar_exhausted"] is False
    assert config["scope"]["trillion_formula_campaign_executed"] is False


def test_seeded_prefix_is_unique_reproducible_and_in_range() -> None:
    config = load_config(ROOT)
    first = _lane_ordinals(config, "pseudorandom")[:10_000]
    replay = _lane_ordinals(config, "pseudorandom")[:10_000]
    assert np.array_equal(first, replay)
    assert len(np.unique(first)) == len(first)
    assert np.all(first < config["program_grammar"]["full_ordinal_space"])
    decoded = decode_ordinals(first, config)
    assert np.max(decoded["left_primitive_index"]) < 440
    assert np.max(decoded["right_primitive_index"]) < 440
    assert np.max(decoded["operator_index"]) < 8


def test_response_blind_primitive_receipt() -> None:
    receipt = build_primitive_receipt(ROOT)
    assert receipt["response_fields_read"] == []
    assert receipt["response_values_used"] == 0
    assert receipt["audit"]["shape"] == [440, 112]
    assert receipt["audit"]["item_counts"] == {
        "45": 64,
        "46": 184,
        "47": 96,
        "48": 96,
    }


def test_behavior_selection_uses_balanced_object_loss() -> None:
    config = load_config(ROOT)
    arrays = {
        "target": np.asarray([1.0, 1.0, 2.0, 2.0]),
        "base": np.zeros(4),
        "sigma": np.ones(4),
        "population": np.asarray(["S4TM", "S4TM", "CLASH", "CLASH"]),
        "object": np.asarray(["g1", "g2", "c1", "c1"]),
    }
    behavior = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, 2.0]])
    row, loss, _backend, count = _best_behavior(
        behavior, arrays, np.ones(4, dtype=bool), config
    )
    assert row == 1
    assert loss == 0.0
    assert count == 8


def test_recorded_freeze_receipts_replay_exactly() -> None:
    config = load_config(ROOT)
    source = ROOT / config["paths"]["source_dir"]
    candidate = json.loads(
        (source / config["paths"]["candidate_manifest"]).read_text(encoding="utf-8")
    )
    assert candidate == build_candidate_manifest(ROOT)
    assert candidate["total_raw_schedule_positions"] == 2_097_152
    assert candidate["lane_audits"]["pseudorandom"][
        "programs_eligible_for_response_scoring"
    ] == 86_561
    assert candidate["lane_audits"]["sequential_ordinal_control"][
        "programs_eligible_for_response_scoring"
    ] == 40_992
    assert candidate["response_values_used_during_program_generation"] == 0
    assert json.loads(
        (source / config["paths"]["primitive_receipt"]).read_text(encoding="utf-8")
    ) == build_primitive_receipt(ROOT)
    assert json.loads(
        (source / config["paths"]["exposure_manifest"]).read_text(encoding="utf-8")
    ) == build_exposure_manifest(ROOT)


def test_recorded_item49_result_is_reproducible_and_not_promoted() -> None:
    config = load_config(ROOT)
    source = ROOT / config["paths"]["source_dir"]
    evaluation = json.loads(
        (source / config["paths"]["evaluation_result"]).read_text(encoding="utf-8")
    )
    aggregate = json.loads(
        (ROOT / config["paths"]["aggregate_result"]).read_text(encoding="utf-8")
    )
    assert evaluation == build_evaluation_result(ROOT)
    assert aggregate == build_aggregate_result(ROOT)
    assert evaluation["counts"]["raw_schedule_positions"] == 2_097_152
    assert evaluation["counts"]["outcome_scored_behavior_classes"] == 127_553
    assert evaluation["compute"]["program_point_fold_evaluations"] == 71_429_680
    assert {
        row["selected_pseudorandom_program"]["ordinal"]
        for row in evaluation["fold_ledger"]
    } == {1_575_724_121_867}
    assert evaluation["scores"]["pseudorandom_program_search"][
        "balanced_loss"
    ] < evaluation["scores"]["sequential_ordinal_control"]["balanced_loss"]
    assert evaluation["scores"]["pseudorandom_program_search"][
        "balanced_loss"
    ] > evaluation["scores"]["item45_universal_interaction"]["balanced_loss"]
    assert aggregate["decision"] == "NONPROMOTED_ITEM49_PSEUDORANDOM_RESULT_RETAINED"
    assert aggregate["claims"]["formula_family_pruned"] is False
    assert aggregate["claims"]["historical_novelty_established"] is False
