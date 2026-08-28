from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    GravityItem22Error,
    _basis,
    _build_sample,
    _candidate_digest,
    _candidate_log_mu,
    _content_hashed,
    _contract_digest,
    _exact_signature_classes,
    _mode_weights,
    _normal_identity,
    _response_url,
    _screen_log_mu,
    _verify_content_hash,
    generate_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_balanced_generator_are_frozen() -> None:
    config = load_config(ROOT)
    arrays = generate_candidates(config)
    assert len(arrays["niche"]) == 262144
    assert np.bincount(arrays["niche"]).tolist() == [65536] * 4
    assert np.bincount(arrays["polarity"]).tolist() == [131072, 131072]
    assert (
        _candidate_digest(arrays)
        == "f1274428d61d762aa62cd828c7dc3785b60c24f61826158bfffa1288e923fa7d"
    )
    assert _exact_signature_classes(config, arrays) == 99860
    for niche in range(4):
        assert np.bincount(arrays["polarity"][arrays["niche"] == niche]).tolist() == [
            32768,
            32768,
        ]
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["discovery_policy"]["equal_initial_viability"] is True


def test_contract_digest_ignores_only_commit_bindings() -> None:
    config = load_config(ROOT)
    rebound = json.loads(json.dumps(config))
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    rebound["gates"]["minimum_joint_mse_improvement_vs_flexible_nuisance"] = -1.0
    assert _contract_digest(config) != _contract_digest(rebound)


def test_identity_normalization_catches_survey_prefix_rewrites() -> None:
    assert _normal_identity("SDSSJ2321-0939") == _normal_identity("J2321-0939")
    assert _normal_identity("SL2SJ021247-055552") == _normal_identity("J021247-055552")


def test_mode_weights_cover_pure_pair_and_triple_niches() -> None:
    values = {
        "niche": np.asarray([0, 1, 2, 3]),
        "pure_mode": np.asarray([1, 0, 0, 0]),
        "pair_mode": np.asarray([0, 2, 0, 0]),
        "pair_mixing": np.asarray([0.2, 0.35, 0.5, 0.5]),
        "triple_simplex": np.asarray(
            [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.6, 0.2, 0.2]]
        ),
    }
    weights = _mode_weights(values, np)
    assert np.allclose(weights[0], [0.0, 1.0, 0.0])
    assert np.allclose(weights[1], [0.0, 0.35, 0.65])
    assert np.allclose(weights[2], [0.2, 0.3, 0.5])
    assert np.allclose(weights[3], [0.6, 0.2, 0.2])
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_scalar_projector_changes_dynamics_but_not_light() -> None:
    config = load_config(ROOT)
    arrays = {
        "niche": np.asarray([0], dtype=np.int16),
        "polarity": np.asarray([1], dtype=np.int16),
        "amplitude": np.asarray([3], dtype=np.int16),
        "lambda_scalar": np.asarray([5], dtype=np.int16),
        "range_vector": np.asarray([0], dtype=np.int16),
        "range_tensor": np.asarray([1], dtype=np.int16),
        "pure_mode": np.asarray([0], dtype=np.int16),
        "pair_mode": np.asarray([0], dtype=np.int16),
        "pair_mixing": np.asarray([0], dtype=np.int16),
        "triple_simplex": np.asarray([0], dtype=np.int16),
        "interaction_pair": np.asarray([0], dtype=np.int16),
        "interaction_fraction": np.asarray([0], dtype=np.int16),
        "interaction_phase": np.asarray([0], dtype=np.int16),
    }
    result = _candidate_log_mu(
        config, arrays, [{"reff_kpc": 1.0, "rein_kpc": 1.0}], 0, 1, np
    )
    expected = float(_basis(np.asarray([1.0]), np)[0])
    assert math.isclose(result[0, 0, 0], expected, rel_tol=1e-12)
    assert result[0, 0, 1] == 0.0


def test_predictor_only_sample_balances_roles_and_folds() -> None:
    config = load_config(ROOT)
    predictors = [
        {
            "name": f"LENS{index:03d}",
            "z_lens": 0.05 + 0.01 * index,
            "survey": "BELLS",
            "z_source": 1.0,
            "z_luminosity_Lsun": 1e11 + index,
            "reff_kpc": 2.0,
            "g_minus_r": 1.0,
            "r_minus_z": 0.5,
            "axis_ratio": 0.8,
            "max_fracflux_grz": 0.01,
        }
        for index in range(50)
    ]
    sample = _build_sample(predictors, config)
    assert sample["counts"] == {
        "predictor_quality_eligible": 50,
        "selected": 50,
        "exploration": 40,
        "confirmation": 10,
    }
    assert sample["fold_counts"] == {str(index): 8 for index in range(5)}
    for stratum in range(5):
        group = [row for row in sample["objects"] if row["redshift_stratum"] == stratum]
        assert len(group) == 10
        assert sum(row["role"] == "confirmation" for row in group) == 2


def test_response_query_reads_only_one_frozen_exploration_row() -> None:
    config = load_config(ROOT)
    url = _response_url(config, "SDSSJ0000+0000")
    assert "Name=SDSSJ0000%2B0000" in url
    assert "thetaE%2Ce_thetaE%2Csigma%2CE_sigma%2Ce_sigma%2Cf_sigma" in url
    assert "Ref" not in url
    assert "Dobs" not in url


def test_small_screen_recovers_linked_two_channel_signal() -> None:
    config = load_config(ROOT)
    folds = np.asarray([index % 5 for index in range(20)])
    x = np.linspace(0.0, 1.0, len(folds))
    log_mu = np.stack(
        [
            np.zeros((len(folds), 2)),
            np.column_stack([0.3 * x, 0.15 * x]),
            np.column_stack([0.05 * x**2, 0.4 * x]),
        ]
    )
    y = math.log(2.0) + log_mu[1]
    result = _screen_log_mu(log_mu, y, folds, config, np)
    assert result["selected_indices"] == [1] * 5
    assert np.max(np.abs(result["prediction"] - y)) < 1e-12


def test_content_hash_detects_mutation() -> None:
    payload = _content_hashed({"a": 1, "b": [2, 3]})
    _verify_content_hash(payload, "test")
    payload["a"] = 2
    with pytest.raises(GravityItem22Error, match="changed"):
        _verify_content_hash(payload, "test")
