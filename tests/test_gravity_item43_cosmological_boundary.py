from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item43_cosmological_boundary import (
    admissible_candidates,
    age_ratio,
    boundary_bases,
    build_candidate_manifest,
    build_exposure_manifest,
    decode_candidate,
    expansion_ratio,
    generate_raw_candidates,
    load_config,
    sersic_n4_fraction,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_counterexample_and_response_boundaries() -> None:
    config = load_config(ROOT)
    assert config["item"] == 43
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ]
    assert config["discovery_policy"]["counterexample_count_alone_is_never_decisive"]
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["schema_audit_exposure"]["rows_with_response_seen"] == 5


def test_raw_grid_has_four_equal_cosmological_niches() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["candidate_id"]) == 262_144
    assert [int(np.sum(raw["lane"] == lane)) for lane in range(4)] == [65_536] * 4
    assert decode_candidate(0, config)["lane"] == "expansion_rate_running"
    assert decode_candidate(196_608, config)["lane"] == "finite_horizon_fraction"


def test_boundary_coordinates_have_present_epoch_and_monotone_limits() -> None:
    config = load_config(ROOT)
    z = np.asarray([0.0, 0.5, 1.0])
    radius = np.asarray([1.0, 10.0, 100.0])
    assert np.all(np.diff(expansion_ratio(z, config)) > 0.0)
    assert np.all(np.diff(age_ratio(z, config)) < 0.0)
    bases = boundary_bases(z, radius, config)
    assert bases.shape == (4, 3)
    assert np.allclose(bases[:3, 0], 1.0)
    assert np.all(bases > 0.0)


def test_de_vaucouleurs_aperture_fraction_is_half_at_re() -> None:
    config = load_config(ROOT)
    b4 = float(config["constants"]["sersic_n4_b"])
    values = sersic_n4_fraction(np.asarray([0.0, 1.0, 10.0]), b4)
    assert values[0] == 0.0
    assert abs(values[1] - 0.5) < 1e-8
    assert 0.9 < values[2] < 1.0


def test_admission_and_freeze_manifests_are_response_safe() -> None:
    config = load_config(ROOT)
    admitted, audit = admissible_candidates(config)
    assert audit["raw_candidates"] == 262_144
    assert audit["admitted_candidates"] == 186_989
    assert len(admitted["candidate_id"]) == audit["admitted_candidates"]
    candidate = build_candidate_manifest(ROOT)
    exposure = build_exposure_manifest(ROOT)
    assert candidate["response_accessed_during_generation"] is False
    assert candidate["confirmation_accessed"] is False
    assert exposure["counts"]["schema_rows_with_response_seen"] == 5
    assert exposure["counts"]["remaining_response_rows_read"] == 0
    assert exposure["counts"]["confirmation_rows_read"] == 0
