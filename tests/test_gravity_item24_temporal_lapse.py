from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item24_temporal_lapse import (
    GravityItem24Error,
    _build_sample,
    _content_hashed,
    _contract_digest,
    _coordinate_from_sdss_name,
    _estimate_delay,
    _mechanism_H,
    _query_url,
    _responses_from_H,
    _verify_content_hash,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_balances_temporal_mechanisms_and_policies() -> None:
    config = load_config(ROOT)
    arrays = generate_raw_candidates(config)
    assert len(arrays["niche"]) == 262144
    assert np.bincount(arrays["niche"]).tolist() == [65536] * 4
    assert config["discovery_policy"]["age_or_history_is_not_privileged"] is True
    assert config["discovery_policy"]["paper_claim_requires_fresh_replication"] is True
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["scope"]["confirmation_opening_authorized"] is False


def test_contract_digest_ignores_only_freeze_commit_bindings() -> None:
    config = load_config(ROOT)
    rebound = json.loads(json.dumps(config))
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    rebound["discovery_policy"]["age_or_history_is_not_privileged"] = False
    assert _contract_digest(config) != _contract_digest(rebound)


def test_one_metric_rule_links_motion_and_photon_response() -> None:
    values = {
        "niche": np.asarray([0]),
        "amplitude": np.asarray([0.3]),
        "polarity": np.asarray([1.0]),
        "transition_acceleration": np.asarray([1e-10]),
        "transition_power": np.asarray([2.0]),
        "slip": np.asarray([1.0]),
        "compactness_threshold": np.asarray([1e-8]),
        "memory_multiplier": np.asarray([1000.0]),
        "resonance_frequency": np.asarray([1.0]),
        "resonance_phase": np.asarray([0.0]),
    }
    h = _mechanism_H(
        values, np.asarray([1e-12]), np.asarray([1e-9]), np.asarray([1e-4]), np
    )
    motion, photon = _responses_from_H(values, h, np)
    assert motion.shape == photon.shape == (1, 1)
    assert math.isclose(float(motion[0, 0]), float(photon[0, 0]))
    values["slip"] = np.asarray([0.0])
    _, photon_half = _responses_from_H(values, h, np)
    assert math.isclose(float(photon_half[0, 0] - 1.0), 0.5 * float(h[0, 0]))


def test_target_query_boundaries_hide_width_and_published_delay() -> None:
    config = load_config(ROOT)
    safe = _query_url(
        "J/ApJ/861/49/table2", config["sources"]["alfalfa_predictor_columns"]
    )
    response = _query_url(
        "J/ApJ/861/49/table2", config["sources"]["alfalfa_response_columns"], AGC=42
    )
    assert "W50" not in safe
    assert "W50" in response
    assert "delay" not in safe.lower()
    assert config["sources"]["published_delay_answers_allowed"] is False


def test_sample_seals_both_lanes_and_balances_folds() -> None:
    config = load_config(ROOT)
    galaxies = []
    for stratum in range(4):
        for index in range(45):
            galaxies.append(
                {
                    "agc": stratum * 1000 + index,
                    "name": f"G{stratum}-{index}",
                    "mass_stratum": stratum,
                }
            )
    lenses = [
        {"name": f"L{index}", "z_lens": 0.1 + 0.05 * index} for index in range(10)
    ]
    sample = _build_sample(galaxies, lenses, config)
    assert sample["counts"] == {
        "galaxy_motion:confirmation": 32,
        "galaxy_motion:exploration": 128,
        "photon_delay:confirmation": 2,
        "photon_delay:exploration": 8,
    }
    assert all(sample["fold_counts"][f"galaxy_motion:{fold}"] == 32 for fold in range(4))
    assert all(sample["fold_counts"][f"photon_delay:{fold}"] == 2 for fold in range(4))


def test_synthetic_raw_delay_estimator_recovers_shift_without_answer_table() -> None:
    config = load_config(ROOT)
    time = np.arange(0.0, 1200.0, 3.0)
    signal = 0.12 * np.sin(time / 37.0) + 0.05 * np.sin(time / 11.0)
    shift = 23.0
    a = signal
    b = 0.12 * np.sin((time - shift) / 37.0) + 0.05 * np.sin((time - shift) / 11.0) + 0.7
    lines = [
        f"{t:.5f} {ma:.5f} 0.00500 {mb:.5f} 0.00500 SYNTH"
        for t, ma, mb in zip(time, a, b, strict=True)
    ]
    estimate = _estimate_delay(("\n".join(lines) + "\n").encode(), config)
    assert abs(estimate["delay_days"] - shift) <= 2.0
    assert estimate["quality_pass"] is True


def test_sdss_coordinate_parser_handles_full_precision_names() -> None:
    coordinate = _coordinate_from_sdss_name("J083216.99+040405.2")
    assert coordinate is not None
    assert coordinate[0] == pytest.approx(128.0707917, abs=1e-6)
    assert coordinate[1] == pytest.approx(4.0681111, abs=1e-6)


def test_content_hash_detects_mutation() -> None:
    payload = _content_hashed({"a": 1, "b": [2, 3]})
    _verify_content_hash(payload, "test")
    payload["a"] = 2
    with pytest.raises(GravityItem24Error, match="changed"):
        _verify_content_hash(payload, "test")
