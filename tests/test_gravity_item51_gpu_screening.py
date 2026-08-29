from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    decode_ordinals,
    load_config as load_item49_config,
)
from sigma_theory_compiler.gravity_item51_gpu_screening import (
    _canonical_symbolic_keys,
    _schedule_ordinals,
    build_preflight_manifest,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_item51_freezes_an_honest_collision_free_67m_schedule() -> None:
    config = load_config(ROOT, require_bound=False)
    schedule = config["schedule"]
    space = config["program_grammar_binding"]["full_ordinal_space"]
    assert schedule["sample_positions"] == 67_108_864
    assert math.gcd(schedule["coprime_stride"], space) == 1
    assert schedule["sample_positions"] < space
    assert config["scope"]["trillion_formula_campaign_executed"] is False
    assert config["scope"]["full_grammar_exhausted"] is False
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ] is True
    assert config["discovery_policy"]["finite_empirical_sample_may_prune_family"] is False


def test_affine_prefix_is_deterministic_unique_and_in_range() -> None:
    config = load_config(ROOT, require_bound=False)
    first = _schedule_ordinals(config, 0, 100_000)
    replay = _schedule_ordinals(config, 0, 100_000)
    assert np.array_equal(first, replay)
    assert len(np.unique(first)) == len(first)
    assert int(np.min(first)) >= 0
    assert int(np.max(first)) < config["program_grammar_binding"]["full_ordinal_space"]


def test_packed_symbolic_key_matches_commutative_equivalence_rule() -> None:
    config49 = load_item49_config(ROOT)
    # weighted_product is commutative for every mixing value in the Item 49 rule.
    left = {
        "ordinal": np.asarray([0, 1], dtype=np.uint64),
        "transition_index": np.asarray([0, 0], dtype=np.int16),
        "exponent_index": np.asarray([0, 0], dtype=np.int16),
        "amplitude_index": np.asarray([9, 9], dtype=np.int16),
        "mixing_index": np.asarray([5, 5], dtype=np.int16),
        "operator_index": np.asarray([2, 2], dtype=np.int16),
        "right_transform_index": np.asarray([3, 1], dtype=np.int16),
        "right_primitive_index": np.asarray([8, 7], dtype=np.int16),
        "left_transform_index": np.asarray([1, 3], dtype=np.int16),
        "left_primitive_index": np.asarray([7, 8], dtype=np.int16),
    }
    keys = _canonical_symbolic_keys(left, config49)
    assert keys[0] == keys[1]


def test_preflight_is_response_blind_before_live_gpu_check() -> None:
    preflight = build_preflight_manifest(ROOT, live=False)
    assert preflight["response_fields_used_to_construct_schedule"] == []
    assert preflight["response_values_used_to_construct_schedule"] == 0
    assert preflight["sealed_confirmation_rows"] == 0
    assert preflight["paid_model_calls"] == 0


def test_schedule_decodes_under_the_bound_item49_grammar() -> None:
    config = load_config(ROOT, require_bound=False)
    config49 = load_item49_config(ROOT)
    ordinals = _schedule_ordinals(config, 1_000_000, 1024)
    decoded = decode_ordinals(ordinals, config49)
    assert np.array_equal(decoded["ordinal"], ordinals)
    assert np.all(decoded["left_primitive_index"] < 440)
    assert np.all(decoded["right_primitive_index"] < 440)
