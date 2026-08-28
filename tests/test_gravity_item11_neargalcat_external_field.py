from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item11_neargalcat_external_field as external

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_fresh_real_source_generator_and_response_boundary() -> None:
    config = external.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 11
    assert config["predecessor"]["required_decision"] == (
        "INCONCLUSIVE_ITEM10_SOURCE_QUALITY_NEGATIVE_DIRECTION_ADVANCE_ITEM11"
    )
    assert config["source"]["observed_rows"] == 869
    assert config["source"]["prefreeze_queries"]["predictor_rows_read"] == 0
    assert config["source"]["prefreeze_queries"]["response_rows_read"] == 0
    assert config["candidate_generator"]["candidate_cells"] == 262144
    assert len(config["candidate_generator"]["families"]) == 12
    assert config["candidate_generator"]["historical_novelty_claimed"] is False
    assert config["source"]["confirmation_query_forbidden"] is True
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["authorization"]["indicative_or_dynamical_mass_features_allowed"] is False


def test_plain_table_parser_and_predictor_derivation() -> None:
    payload = b"a | b\n 1 | hello \n 2 | world\n\nNumber of rows: 2\n"
    assert external._parse_plain_table(payload) == [
        {"a": "1", "b": "hello"},
        {"a": "2", "b": "world"},
    ]
    config = external.load_config(ROOT)
    row = {
        "__row": "1",
        "name": "NGC 0001",
        "ra": "10",
        "dec": "-20",
        "bmag": "12",
        "ks_mag": "8",
        "linear_diameter": "8",
        "distance": "5",
        "inclination": "60",
        "log_h1_mass": "8.5",
        "log_h1_mass_limit": "null",
        "neighbor_galaxy_name": "NGC 2",
        "morph_type": "8",
        "tidal_index_1": "-0.5",
        "tidal_index_2": "0.1",
        "axial_ratio": "0.6",
        "log_ks_lum_density": "0.2",
        "bmag_surface_brightness": "23",
        "log_ks_luminosity": "9",
    }
    derived = external.derive_predictors(row, config)
    assert derived["normalized_identity"] == "NGC1"
    assert 0 < derived["gas_fraction"] < 1
    assert derived["log_baryonic_mass"] > derived["log_stellar_mass"]
    assert derived["inclination_sine"] == pytest.approx(np.sin(np.deg2rad(60)))
    assert derived["b_minus_ks_color"] == 4


def test_access_requires_exact_commit_bindings() -> None:
    if external.SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(external.GravityItem11ExternalFieldError, match="not bound"):
            external.write_predictor_source(ROOT)
    if external.SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(external.GravityItem11ExternalFieldError, match="not bound"):
            external.write_response_source(ROOT)


def test_predecessor_exclusion_registry_is_nonempty_and_normalized() -> None:
    config = external.load_config(ROOT)
    identities = external._predecessor_identities(ROOT, config)
    coordinates = external._predecessor_coordinates(ROOT, config)
    assert len(identities) > 1000
    assert coordinates.shape[0] > 1000
    assert coordinates.shape[1] == 2
    assert external.normalize_identity("NAME NGC 0001") == "NGC1"
    assert all(identity == external.normalize_identity(identity) for identity in identities)


def test_pseudorandom_generator_is_deterministic_and_fully_labeled() -> None:
    config = external.load_config(ROOT)
    first = external.generate_candidates(config)
    second = external.generate_candidates(config)
    assert external._candidate_digest(first) == external._candidate_digest(second)
    assert len(first["family"]) == 262144
    assert all(np.array_equal(first[key], second[key]) for key in first)
    manifest = external.build_candidate_manifest(ROOT)
    assert sum(manifest["family_counts"].values()) == 262144
    assert sum(manifest["origin_status_counts"].values()) == 262144
    assert manifest["counts"]["post_response_cells"] == 0
    assert manifest["counts"]["response_rows_read"] == 0
    assert manifest["claims"]["historical_novelty_established"] is False


def test_all_external_field_families_are_finite_and_distinct() -> None:
    candidates = {
        "family": np.arange(12, dtype=np.int16),
        "threshold": np.linspace(-1.1, 1.1, 12),
        "scale": np.linspace(0.2, 1.3, 12),
        "power": np.linspace(0.5, 3.0, 12),
        "phase": np.linspace(0.1, 2.0, 12),
        "modulation": np.zeros(12, dtype=np.int8),
    }
    data = {
        "theta1": np.linspace(-2, 2, 13),
        "theta2": np.linspace(1.7, -1.3, 13),
        "rho_k": np.linspace(-1.1, 1.9, 13),
        "log_g": np.linspace(1.0, 4.0, 13),
        "surface": np.linspace(6, 9, 13),
        "gas_fraction": np.linspace(0.05, 0.9, 13),
        "size": np.linspace(-0.5, 1.5, 13),
    }
    components = external._candidate_components(candidates, data, 0, 12, np)
    assert components.shape == (12, 13)
    assert np.all(np.isfinite(components))
    assert len({np.round(row, 10).tobytes() for row in components}) == 12


def test_nested_selector_executes_frozen_fold_structure_on_synthetic_data() -> None:
    config = copy.deepcopy(external.load_config(ROOT))
    config["candidate_generator"]["candidate_cells"] = 96
    config["evaluation"]["candidate_batch_size"] = 24
    config["evaluation"]["cpu_crosscheck_candidates"] = 8
    count = 20
    random = np.random.default_rng(11)
    data = {
        "folds": np.arange(count) % 5,
        "y": 1.7 + random.normal(0, 0.05, count),
        "design": random.normal(size=(count, 11)),
        "theta1": np.linspace(-2, 2, count),
        "theta2": random.normal(size=count),
        "rho_k": random.normal(size=count),
        "log_g": np.linspace(1, 4, count),
        "surface": random.normal(size=count),
        "gas_fraction": np.linspace(0.05, 0.9, count),
        "size": random.normal(size=count),
    }
    baseline, proposed, selections, compute = external._nested_select(data, config)
    assert np.all(np.isfinite(baseline))
    assert np.all(np.isfinite(proposed))
    assert len(selections) == 5
    assert compute["candidate_cells"] == 96
    assert compute["candidate_galaxy_score_evaluations"] == 96 * count * 20


def test_formula_and_sample_builders_have_no_response_parameter() -> None:
    for builder in (
        external.derive_predictors,
        external.generate_candidates,
        external.build_candidate_manifest,
        external.build_sample_manifest,
    ):
        signature = " ".join(inspect.signature(builder).parameters).lower()
        assert "velocity" not in signature
        assert "response" not in signature
    source = inspect.getsource(external.write_response_source)
    assert 'row["role"] == "exploration"' in source
    assert 'row["role"] == "reserved_confirmation"' in source
    assert 'WHERE "__row" IN' in source


def test_stored_artifacts_replay_if_present() -> None:
    config = external.load_config(ROOT)
    validators = (
        ("predictor_source", external.validate_predictor_source),
        ("sample_manifest", external.validate_sample_manifest),
        ("candidate_manifest", external.validate_candidate_manifest),
        ("response_source", external.validate_response_source),
    )
    for key, validator in validators:
        path = ROOT / config["outputs"][key]
        if path.exists():
            validator(json.loads(path.read_text(encoding="utf-8")), ROOT)


def test_stored_result_replays_if_present() -> None:
    config = external.load_config(ROOT)
    path = ROOT / config["outputs"]["result"]
    if not path.exists():
        pytest.skip("NEARGALCAT external-field exploration has not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    external.validate_receipt(stored, ROOT)
    external.check_receipt(ROOT)
    assert stored["counts"]["candidate_cells"] == 262144
    assert stored["counts"]["confirmation_response_rows"] == 0
    assert stored["counts"]["post_response_formula_cells"] == 0
    assert stored["counts"]["paid_model_calls"] == 0
