"""Controls for the target-blind baryonic-structure G1 pilot repair."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g0_experiment import load_config as load_g0_config
from sigma_theory_compiler.gravity_g1_pilot_v3 import (
    COMPONENT_COUNT,
    FEATURE_IDS,
    OUTPUT_PATH,
    WIDTHS,
    GravityG1PilotV3Error,
    _center_values,
    baryonic_features,
    build_receipt,
    lexicographic_pair_batches,
    load_config,
    replay_candidate,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.sparc_full_sample import assemble

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def _population() -> Any:
    if "population" not in _CACHE:
        _CACHE["population"] = assemble(ROOT)
    return _CACHE["population"]


def _component_id(feature: str, center: float, width: float) -> int:
    feature_index = FEATURE_IDS.index(feature)
    center_rank = int(np.argmin(np.abs(_center_values(feature) - center)))
    width_index = int(np.argmin(np.abs(WIDTHS - width)))
    return feature_index * 1024 + center_rank * 16 + width_index * 2


def _ordinal(first: tuple[str, float, float], second: tuple[str, float, float]) -> int:
    components = sorted((_component_id(*first), _component_id(*second)))
    return components[0] * COMPONENT_COUNT + components[1]


def test_v3_is_bound_to_the_sealed_ten_of_twelve_predecessor() -> None:
    config = _config()
    assert config["predecessor_binding"]["required_covered_galaxies"] == 10
    assert config["predecessor_binding"]["required_uncovered_galaxies"] == [
        "UGC11820",
        "UGC11455",
    ]
    assert config["repair_galaxies"] == ["UGC11820", "UGC11455"]
    assert [item["id"] for item in config["target_blind_features"]] == list(FEATURE_IDS)
    assert config["candidate_shell"]["maximum_local_constants"] == 2


def test_baryonic_features_do_not_read_velocity_targets_or_errors() -> None:
    galaxy = next(item for item in _population().exploration if item.name == "UGC11820")
    before = baryonic_features(galaxy, 3702.81458)
    mutated = replace(
        galaxy,
        v_obs=tuple(value * 1000 for value in galaxy.v_obs),
        e_v_obs=tuple(value / 1000 for value in galaxy.e_v_obs),
    )
    after = baryonic_features(mutated, 3702.81458)
    assert set(before) == set(after) == set(FEATURE_IDS)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_structured_feature_pair_schedule_is_unique_and_canonical() -> None:
    ordinals = [
        int(value)
        for batch in lexicographic_pair_batches(100_000, 65_536)
        for value in batch
    ]
    assert len(ordinals) == len(set(ordinals)) == 100_000
    assert all(value // COMPONENT_COUNT < value % COMPONENT_COUNT for value in ordinals)


@pytest.mark.parametrize(
    ("galaxy_name", "first", "second"),
    [
        (
            "UGC11820",
            ("log_y", 5.968, 2.0),
            ("gas_fraction", 0.2565, 1.0),
        ),
        (
            "UGC11455",
            ("log_r_over_disk_peak", -0.8888889, 1.0),
            ("gas_to_disk", -2.0385, 1.0),
        ),
    ],
)
def test_counterexample_guided_feature_pair_clears_every_frozen_fold(
    galaxy_name: str,
    first: tuple[str, float, float],
    second: tuple[str, float, float],
) -> None:
    galaxy = next(item for item in _population().exploration if item.name == galaxy_name)
    result = replay_candidate(
        galaxy,
        "structured_occam",
        _ordinal(first, second),
        _config(),
        load_g0_config(ROOT),
    )
    assert result["admitted"] is True
    assert result["failure_obligations"] == []
    assert all(all(row["checks"].values()) for row in result["folds"])
    assert result["description_length"]["local_constant_bits"] == 128


def test_small_cpu_v3_run_cannot_pass_the_full_budget_gate() -> None:
    receipt = build_receipt(ROOT, candidate_count=32, use_gpu=False)
    assert receipt["decision"] == "BLOCK_G1_PILOT_V3_REPAIR_UNCOVERED_OR_INCOMPLETE"
    assert receipt["counts"]["new_v3_candidate_galaxy_trials"] == 192
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0


def test_checked_v3_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 v3 repair has not run yet")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root=ROOT)


def test_v3_receipt_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 v3 repair has not run yet")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["counts"]["confirmation_evaluator_accesses"] = 1
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG1PilotV3Error, match="confirmation access"):
        validate_receipt(tampered, root=ROOT)
