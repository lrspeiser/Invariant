from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item42_matter_geometry_feedback import (
    admissible_candidates,
    build_candidate_manifest,
    build_exposure_manifest,
    decode_candidate,
    feedback_library,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_counterexample_and_response_boundaries() -> None:
    config = load_config(ROOT)
    assert config["item"] == 42
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ]
    assert config["discovery_policy"]["counterexample_count_alone_is_never_decisive"]
    assert config["candidate_generator"]["post_response_cells"] == 0


def test_raw_grid_has_four_equal_feedback_niches() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["candidate_id"]) == 262_144
    assert [int(np.sum(raw["lane"] == lane)) for lane in range(4)] == [65_536] * 4
    assert decode_candidate(0, config)["lane"] == "curvature_matter_reinforcement"
    assert decode_candidate(196_608, config)["lane"] == (
        "two_channel_geometry_competition"
    )


def test_fixed_point_is_response_blind_convergent_and_material() -> None:
    config = load_config(ROOT)
    radius = np.linspace(0.1, 1.0, 10)
    cumulative = np.cumsum(np.exp(-3.0 * radius))
    library, audit = feedback_library(radius, cumulative, config)
    assert library.shape == (4, 16, 10)
    assert audit["all_converged"] is True
    assert np.allclose(library[:, 0], library[0, 0])
    assert np.max(np.abs(library[0, -1] - library[0, 0])) > 0.01
    assert np.max(np.abs(library[1, -1] - library[1, 0])) > 0.01


def test_admission_and_manifests_are_response_safe() -> None:
    config = load_config(ROOT)
    admitted, audit = admissible_candidates(config)
    assert audit["raw_candidates"] == 262_144
    assert audit["admitted_candidates"] == 165_242
    assert len(admitted["candidate_id"]) == audit["admitted_candidates"]
    candidate = build_candidate_manifest(ROOT)
    exposure = build_exposure_manifest(ROOT)
    assert candidate["response_accessed_during_generation"] is False
    assert candidate["confirmation_accessed"] is False
    assert exposure["counts"]["excluded_prior_role_records"] == 89
    assert exposure["counts"]["response_values_read_while_building"] == 0
