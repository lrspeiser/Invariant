from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item44_scale_hierarchy import (
    admissible_candidates,
    build_candidate_manifest,
    build_exposure_manifest,
    build_joint_features,
    decode_candidate,
    generate_raw_candidates,
    hierarchy_coordinates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_exposure_and_counterexample_boundaries() -> None:
    config = load_config(ROOT)
    assert config["item"] == 44
    assert config["scope"]["all_empirical_responses_already_exposed"] is True
    assert config["scope"]["fresh_confirmation_claim_allowed"] is False
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ]
    assert config["discovery_policy"]["counterexample_count_alone_is_never_decisive"]
    assert config["candidate_generator"]["post_evaluation_cells"] == 0


def test_raw_grid_has_four_equal_hierarchy_niches() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["candidate_id"]) == 262_144
    assert [int(np.sum(raw["lane"] == lane)) for lane in range(4)] == [65_536] * 4
    assert decode_candidate(0, config)["lane"] == "baryonic_size_transition_match"
    assert decode_candidate(196_608, config)["lane"] == "four_scale_closure"


def test_hierarchy_coordinates_are_finite_dimensionless_and_material() -> None:
    config = load_config(ROOT)
    coordinate, scales = hierarchy_coordinates(
        np.asarray([5.0, 100.0]),
        np.asarray([3.0, 300.0]),
        np.asarray([1e11, 1e13]),
        np.asarray([0.1, 0.5]),
        config,
    )
    assert coordinate.shape == (4, 2)
    assert np.all(np.isfinite(coordinate))
    assert np.all((coordinate > 0.0) & (coordinate <= 1.0))
    assert np.all(scales["transition_length_kpc"] > 0.0)
    assert np.all(scales["curvature_radius_kpc"] > 0.0)
    assert not np.allclose(coordinate[:, 0], coordinate[:, 1])


def test_admission_and_exposure_manifests_are_frozen() -> None:
    config = load_config(ROOT)
    admitted, audit = admissible_candidates(config)
    assert audit["raw_candidates"] == 262_144
    assert audit["admitted_candidates"] == 201_828
    assert len(admitted["candidate_id"]) == audit["admitted_candidates"]
    assert list(audit["admitted_by_lane"].values()) == [50_457] * 4
    candidate = build_candidate_manifest(ROOT)
    exposure = build_exposure_manifest(ROOT)
    assert candidate["response_values_used_during_formula_generation"] == 0
    assert candidate["confirmation_accessed"] is False
    assert exposure["sealed_data"]["item43_s4tm_confirmation_lenses"] == 7
    assert exposure["sealed_data"]["response_rows_read"] == 0


def test_joint_feature_receipt_preserves_whole_objects_and_sealed_data() -> None:
    receipt = build_joint_features(ROOT)
    assert receipt["counts"]["s4tm_lenses"] == 28
    assert receipt["counts"]["clash_clusters"] == 20
    assert receipt["counts"]["clash_points"] == 84
    assert receipt["counts"]["total_points"] == 112
    assert receipt["counts"]["sealed_confirmation_rows"] == 0
    rows = receipt["records"]
    assert len({row["object"] for row in rows if row["population"] == "S4TM"}) == 28
    assert len({row["object"] for row in rows if row["population"] == "CLASH"}) == 20
    assert all(len(row["hierarchy"]) == 4 for row in rows)
