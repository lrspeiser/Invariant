from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item10_wallaby_boundaries as boundaries

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_real_source_generator_and_response_boundary() -> None:
    config = boundaries.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 10
    assert config["predecessor"]["required_decision"] == (
        "REJECT_ITEM9_TESTED_STELLAR_LIGHT_OCCUPANCY_ADVANCE_ITEM10"
    )
    assert config["source"]["observed_rows"] == 303
    assert config["candidate_generator"]["candidate_cells"] == 131072
    assert len(config["candidate_generator"]["families"]) == 12
    assert config["candidate_generator"]["historical_novelty_claimed"] is False
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["source"]["confirmation_query_forbidden"] is True
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["prefreeze_audit"]["response_rows_read"] == 1
    assert config["prefreeze_audit"]["conservatively_excluded_exposed_identity"] == (
        "WALLABY J101655-485238"
    )


def test_synthetic_profile_measurement_finds_a_target_blind_hi_edge() -> None:
    config = boundaries.load_config(ROOT)
    row = {
        "name": "synthetic",
        "ra": "10",
        "dec": "-30",
        "freq": "1400000000",
        "team_release": "test",
        "team_release_kin": "test",
        "Rad_SD": "5,10,15,20,25,30,35,40",
        "SD_model": "8,6,4,2.5,1.5,0.8,0.4,0.2",
        "e_SD_model": "0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1",
    }
    profile = boundaries.measure_profile(row, config)
    edge = profile["boundaries"]["q1"]
    assert edge is not None
    assert profile["radius_kpc"][4] < edge["edge_radius_kpc"] < profile["radius_kpc"][5]
    assert edge["edge_sharpness"] > 0
    assert 0 <= edge["outer_mass_fraction"] <= 1
    assert profile["total_profile_mass"] > 0


def test_predictor_and_response_access_require_their_exact_commit_bindings() -> None:
    if boundaries.SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(boundaries.GravityItem10BoundaryError, match="not bound"):
            boundaries.write_predictor_source(ROOT)
    if boundaries.SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(boundaries.GravityItem10BoundaryError, match="not bound"):
            boundaries.write_response_source(ROOT)


def test_pseudorandom_generator_is_deterministic_and_fully_labeled() -> None:
    config = boundaries.load_config(ROOT)
    first = boundaries.generate_candidates(config)
    second = boundaries.generate_candidates(config)
    assert boundaries._candidate_digest(first) == boundaries._candidate_digest(second)
    assert len(first["family"]) == 131072
    assert all(np.array_equal(first[key], second[key]) for key in first)
    manifest = boundaries.build_candidate_manifest(ROOT)
    assert sum(manifest["family_counts"].values()) == 131072
    assert sum(manifest["origin_status_counts"].values()) == 131072
    assert manifest["counts"]["polarity_equivalence_duplicates_generated"] == 0
    assert manifest["counts"]["response_rows_read"] == 0
    assert manifest["claims"]["historical_novelty_established"] is False


def test_all_twelve_boundary_families_are_finite_and_structurally_distinct() -> None:
    candidates = {
        "family": np.arange(12, dtype=np.int16),
        "threshold": np.arange(12, dtype=np.int8) % 5,
        "scale": np.linspace(0.12, 0.7, 12),
        "power": np.linspace(0.7, 3.0, 12),
        "phase": np.linspace(0.1, 2.1, 12),
        "modulation": np.zeros(12, dtype=np.int8),
    }
    x = np.vstack([np.linspace(0.2 + index * 0.03, 2.0, 11) for index in range(5)])
    data = {
        "x_thresholds": x,
        "valid_thresholds": np.ones_like(x),
        "sharpness_mod": np.full_like(x, 0.4),
        "outer_mass_mod": np.full_like(x, -0.2),
        "concentration_mod": np.linspace(-0.5, 0.5, x.shape[1]),
        "contrast_mod": np.linspace(0.7, -0.7, x.shape[1]),
        "enclosed": np.linspace(0.05, 0.95, x.shape[1]),
    }
    components = boundaries._candidate_components(candidates, data, 0, 12, np)
    assert components.shape == (12, 11)
    assert np.all(np.isfinite(components))
    assert len({np.round(row, 10).tobytes() for row in components}) == 12


def test_nested_selector_executes_all_outer_and_inner_folds_on_synthetic_data() -> None:
    config = copy.deepcopy(boundaries.load_config(ROOT))
    config["candidate_generator"]["candidate_cells"] = 96
    config["evaluation"]["candidate_batch_size"] = 24
    config["evaluation"]["cpu_crosscheck_candidates"] = 8
    galaxy_count = 10
    points_per_galaxy = 3
    point_count = galaxy_count * points_per_galaxy
    galaxy_index = np.repeat(np.arange(galaxy_count), points_per_galaxy)
    folds = np.arange(galaxy_count) % 5
    radius = np.tile(np.asarray([0.45, 0.9, 1.6]), galaxy_count)
    random = np.random.default_rng(10)
    data = {
        "folds": folds,
        "galaxy_index": galaxy_index,
        "point_counts": np.full(galaxy_count, points_per_galaxy),
        "y": 1.7 + 0.11 * np.log10(radius) + random.normal(0, 0.003, point_count),
        "design": random.normal(size=(point_count, 10)),
        "x_thresholds": np.vstack([radius * (1 + 0.08 * index) for index in range(5)]),
        "valid_thresholds": np.ones((5, point_count)),
        "sharpness_mod": np.full((5, point_count), 0.3),
        "outer_mass_mod": np.full((5, point_count), -0.1),
        "concentration_mod": np.repeat(np.linspace(-0.4, 0.4, galaxy_count), 3),
        "contrast_mod": np.repeat(np.linspace(0.5, -0.5, galaxy_count), 3),
        "enclosed": np.tile(np.asarray([0.15, 0.55, 0.9]), galaxy_count),
    }
    baseline, proposed, selections, compute = boundaries._nested_select(data, config)
    assert np.all(np.isfinite(baseline))
    assert np.all(np.isfinite(proposed))
    assert len(selections) == 5
    assert {row["outer_fold"] for row in selections} == set(range(5))
    assert compute["candidate_cells"] == 96
    assert compute["candidate_point_score_evaluations"] == 96 * point_count * 20


def test_formula_and_predictor_builders_have_no_velocity_target_parameter() -> None:
    for builder in (
        boundaries.measure_profile,
        boundaries.generate_candidates,
        boundaries.build_candidate_manifest,
        boundaries.build_sample_manifest,
    ):
        signature = " ".join(inspect.signature(builder).parameters).lower()
        assert "velocity" not in signature
        assert "response" not in signature
    response_scope = inspect.getsource(boundaries._response_scope)
    assert 'rows[0]["role"] == "exploration"' in response_scope
    assert 'rows[0]["role"] == "reserved_confirmation"' in response_scope
    response_source = inspect.getsource(boundaries.write_response_source)
    assert 'scope["retained_exploration"]' in response_source
    assert 'scope["retained_confirmation"]' in response_source


def test_response_scope_conservatively_drops_multi_release_names() -> None:
    config = boundaries.load_config(ROOT)
    path = ROOT / config["outputs"]["sample_manifest"]
    predictor_path = ROOT / config["outputs"]["predictor_source"]
    if not path.exists():
        pytest.skip("WALLABY sample has not been frozen")
    sample = json.loads(path.read_text(encoding="utf-8"))
    predictor = json.loads(predictor_path.read_text(encoding="utf-8"))
    scope = boundaries._response_scope(sample, predictor)
    assert len(scope["retained_exploration"]) == 38
    assert len(scope["retained_confirmation"]) == 11
    assert len(scope["ambiguous_names"]) == 20
    assert scope["ambiguous_release_rows"] == 36
    assert len(scope["catalogue_duplicate_names"]) == 55
    assert scope["initial_attempted_unique_names"] == 55
    assert scope["scope_incident_potential_confirmation_rows"] == 2
    assert not (
        {row["name"] for row in scope["retained_exploration"]}
        & {row["name"] for row in scope["retained_confirmation"]}
    )


def test_stored_prefreeze_artifacts_replay_if_present() -> None:
    config = boundaries.load_config(ROOT)
    predictor_path = ROOT / config["outputs"]["predictor_source"]
    if predictor_path.exists():
        predictor = json.loads(predictor_path.read_text(encoding="utf-8"))
        boundaries.validate_predictor_source(predictor, ROOT)
    sample_path = ROOT / config["outputs"]["sample_manifest"]
    if sample_path.exists():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        boundaries.validate_sample_manifest(sample, ROOT)
    candidate_path = ROOT / config["outputs"]["candidate_manifest"]
    if candidate_path.exists():
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        boundaries.validate_candidate_manifest(candidates, ROOT)


def test_stored_result_replays_if_present() -> None:
    config = boundaries.load_config(ROOT)
    path = ROOT / config["outputs"]["result"]
    if not path.exists():
        pytest.skip("WALLABY boundary exploration has not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    boundaries.validate_receipt(stored, ROOT)
    boundaries.check_receipt(ROOT)
    assert stored["counts"]["candidate_cells"] == 131072
    assert stored["counts"]["stored_confirmation_response_rows"] == 0
    assert (
        bool(stored["counts"]["confirmation_response_rows"])
        == stored["claims"]["confirmation_opened"]
    )
    assert stored["counts"]["post_response_formula_cells"] == 0
    assert stored["counts"]["paid_model_calls"] == 0
