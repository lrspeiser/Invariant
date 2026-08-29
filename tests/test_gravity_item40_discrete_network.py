from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item40_discrete_network import (
    GravityItem40Error,
    admissible_candidates,
    build_candidate_manifest,
    build_exposure_manifest,
    build_predictor_receipt,
    build_sample_manifest,
    decode_candidate,
    graph_coordinates,
    load_config,
    predict_multiplier,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_creativity_and_counterexample_boundaries() -> None:
    config = load_config(ROOT)
    assert config["item"] == 40
    assert config["candidate_generator"]["raw_candidate_cells"] == 262144
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ]
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["cluster_transfer"]["retuning_allowed"] is False


def test_graph_coordinates_are_deterministic_bounded_and_response_free() -> None:
    radius = np.asarray([0.5, 1.0, 1.8, 3.0, 4.5, 6.5, 9.0])
    cumulative = np.asarray([1.0, 2.4, 4.5, 7.2, 10.0, 12.0, 13.0])
    first = graph_coordinates(radius, cumulative)
    second = graph_coordinates(radius, cumulative)
    assert first.shape == (4, len(radius))
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))
    assert first[1, 0] == 0.0
    assert first[1, -1] == pytest.approx(1.0)


def test_graph_coordinates_reject_nonmonotone_source_profile() -> None:
    with pytest.raises(GravityItem40Error):
        graph_coordinates(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 0.5, 2.0]))


def test_candidate_grid_and_admission_are_frozen() -> None:
    config = load_config(ROOT)
    admitted, audit = admissible_candidates(config)
    assert audit["raw_candidates"] == 262144
    assert audit["admitted_candidates"] == 180436
    assert len(admitted["candidate_id"]) == 180436
    assert sum(audit["admitted_by_lane"].values()) == 180436
    decoded = decode_candidate(0, config)
    assert decoded["lane"] == "spectral_fiedler_centrality"
    assert decoded["creativity_label"].startswith("known_graph_spectral")


def test_multiplier_changes_with_graph_coordinate_and_has_local_limit() -> None:
    config = load_config(ROOT)
    candidates = {
        "lane": np.asarray([0], dtype=np.int8),
        "amplitude_index": np.asarray([4], dtype=np.int16),
        "exponent_index": np.asarray([6], dtype=np.int16),
        "transition_index": np.asarray([15], dtype=np.int16),
        "shape_index": np.asarray([6], dtype=np.int16),
    }
    u = np.asarray([1e-3, 1e-3, 1e8])
    features = np.full((4, 3), 0.5)
    features[0, :2] = [0.1, 0.9]
    values = predict_multiplier(candidates, u, features, config)[0]
    assert values[1] > values[0] > 1.0
    assert np.log10(values[-1]) < 1e-3


def test_freeze_manifests_are_response_blind() -> None:
    candidate = build_candidate_manifest(ROOT)
    exposure = build_exposure_manifest(ROOT)
    assert candidate["response_accessed"] is False
    assert candidate["paid_api_calls"] == 0
    assert exposure["counts"]["excluded_item39_identities"] == 75
    assert exposure["counts"]["response_values_read_while_building"] == 0


def test_predictor_receipt_is_response_blind_and_sample_waits_for_binding() -> None:
    receipt = build_predictor_receipt(ROOT)
    assert receipt["counts"]["unused_response_blind_predictors"] == 60
    assert receipt["counts"]["quality_eligible"] == 14
    assert receipt["counts"]["response_rows_read"] == 0
    with pytest.raises(GravityItem40Error, match="predictor freeze"):
        build_sample_manifest(ROOT)
