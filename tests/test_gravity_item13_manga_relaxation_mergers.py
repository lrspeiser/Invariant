from __future__ import annotations

import copy
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item13_manga_relaxation_mergers as merger

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_real_source_age_lead_and_fresh_response_boundary() -> None:
    config = merger.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 13
    assert config["predecessor"]["required_decision"] == (
        "PASS_ITEM12_EXPLORATION_LEAD_RETAINED_ADVANCE_ITEM13"
    )
    morphology = config["sources"]["morphology"]
    assert morphology["observed_rows"] == 10126
    assert morphology["file_bytes"] == 1065600
    assert morphology["file_sha256"] == (
        "0feede0bde8d8224220140d700edbfe7385c18838fdf997de05fcb462d716eaf"
    )
    assert morphology["prefreeze_access"]["morphology_row_values_read"] == 0
    assert morphology["prefreeze_access"]["dynamical_response_rows_read"] == 0
    assert len(config["prior_age_lead"]["cells"]) == 5
    assert config["prior_age_lead"]["post_item12_response_retuning"] is False
    assert config["candidate_generator"]["candidate_cells"] == 262144
    assert len(config["candidate_generator"]["families"]) == 12
    assert config["sources"]["response"]["confirmation_query_forbidden"] is True
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["candidate_generator"]["equivalence_boundaries"]


def test_item12_clock_and_item13_modulation_normalizations_are_frozen() -> None:
    config = merger.load_config(ROOT)
    item12_binding = config["sources"]["item12_config"]
    item12_config = json.loads((ROOT / item12_binding["path"]).read_text(encoding="utf-8"))
    assert merger._sha256_file(ROOT / item12_binding["path"]) == item12_binding["file_sha256"]
    assert (
        config["prior_age_lead"]["fixed_clock_normalization"]
        == item12_config["evaluation"]["fixed_clock_normalization"]
    )
    assert config["evaluation"]["fixed_modulation_normalization"] == {
        "stellar_surface_density": [8.0, 1.5],
        "stellar_mass": [10.5, 1.0],
        "axis_ratio": [0.6, 0.25],
    }


def _item12_row() -> dict[str, object]:
    return {
        "plateifu": "1000-12701",
        "mangaid": "1-100",
        "ra": "150",
        "dec": "2",
        "dn4000": "1.6",
        "d4000": "1.7",
        "hdelta_a": "3",
        "hgamma_a": "1",
        "hbeta": "2",
        "log_surface_density": "8.5",
        "log_stellar_mass": "10",
        "log_half_light_radius": "0.7",
        "axis_ratio": "0.7",
        "sersic_index": "2",
        "g_minus_r_color": "1",
        "redshift": "0.03",
        "log_surface_brightness": "0.7",
        "log_snr": "1.3",
        "snr_med_g": "20",
    }


def _morphology_row() -> dict[str, object]:
    return {
        "Name": "manga-1000-12701",
        "plateifu": "1000-12701",
        "MANGAID": "1-100",
        "objra": 150.0,
        "objdec": 2.0,
        "Type": "Sb",
        "TType": 3,
        "Unsure": 0,
        "Bars": 0.5,
        "Edge_on": 0,
        "Tidal": 1,
        "C": 3.2,
        "E_C": 0.1,
        "A": 0.2,
        "E_A": 0.03,
        "S": 0.15,
        "E_S": 0.04,
        "cas_flag": 1,
    }


def test_predictor_derivation_uses_fixed_morphology_and_prior_age_formula() -> None:
    config = merger.load_config(ROOT)
    derived = merger.derive_predictors(_morphology_row(), _item12_row(), config)
    assert derived["tidal"] == 1
    assert derived["tidal_signed"] == 1
    assert derived["merger_unclassified"] == 0
    assert derived["concentration_normalized"] == pytest.approx(0.2)
    assert derived["asymmetry_normalized"] == pytest.approx(2 / 3)
    assert -1 <= derived["prior_age_lead"] <= 1
    assert derived["prior_age_lead"] == pytest.approx(
        merger._prior_age_component(_item12_row(), config)
    )


def test_cas_quality_and_identity_are_target_blind_exclusions() -> None:
    config = merger.load_config(ROOT)
    morphology = _morphology_row()
    morphology["cas_flag"] = 0
    with pytest.raises(merger.GravityItem13RelaxationError, match="CAS"):
        merger.derive_predictors(morphology, _item12_row(), config)
    morphology = _morphology_row()
    morphology["MANGAID"] = "1-999"
    with pytest.raises(merger.GravityItem13RelaxationError, match="identity"):
        merger.derive_predictors(morphology, _item12_row(), config)


def test_access_requires_exact_commit_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(merger, "SCIENTIFIC_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(merger.GravityItem13RelaxationError, match="not bound"):
        merger.write_predictor_source(ROOT)
    monkeypatch.setattr(merger, "SAMPLE_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(merger.GravityItem13RelaxationError, match="not bound"):
        merger.write_response_source(ROOT)


def test_response_receipt_requires_scientific_and_sample_commit_bindings() -> None:
    source = merger._content_hashed(
        {
            "scientific_freeze_commit": merger.SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": merger.SAMPLE_FREEZE_COMMIT,
            "counts": {
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
            },
        }
    )
    merger.validate_response_source(source, ROOT)
    wrong_science = merger._content_hashed(
        {
            **{key: value for key, value in source.items() if key != "content_sha256"},
            "scientific_freeze_commit": "wrong-scientific-freeze",
        }
    )
    with pytest.raises(merger.GravityItem13RelaxationError, match="scientific"):
        merger.validate_response_source(wrong_science, ROOT)
    wrong_sample = merger._content_hashed(
        {
            **{key: value for key, value in source.items() if key != "content_sha256"},
            "sample_freeze_commit": "wrong-sample-freeze",
        }
    )
    with pytest.raises(merger.GravityItem13RelaxationError, match="sample"):
        merger.validate_response_source(wrong_sample, ROOT)


def test_predecessor_registry_excludes_item12_item2_and_coordinates() -> None:
    config = merger.load_config(ROOT)
    plates, manga = merger._excluded_identities(ROOT, config)
    coordinates = merger._coordinates(ROOT, config)
    assert len(plates) >= 1090
    assert len(manga) >= 1090
    assert coordinates.shape[0] > 1000
    assert coordinates.shape[1] == 2


def test_target_blind_manifest_balances_tidal_mass_cells_and_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(merger.load_config(ROOT))
    config["outputs"]["predictor_source"] = "predictor-source.json"
    records = []
    ordinal = 0
    for tidal, mass in ((0, 9.0), (0, 11.0), (1, 9.0), (1, 11.0)):
        for _ in range(120):
            ordinal += 1
            records.append(
                {
                    "plateifu": f"{ordinal}-1",
                    "mangaid": f"1-{ordinal}",
                    "ra": "10",
                    "dec": "10",
                    "log_stellar_mass": merger._metric(mass),
                    "tidal": tidal,
                }
            )
    source = merger._content_hashed({"records": records})
    (tmp_path / "predictor-source.json").write_bytes(merger.canonical_json_bytes(source) + b"\n")
    monkeypatch.setattr(merger, "load_config", lambda root: config)
    monkeypatch.setattr(merger, "validate_predictor_source", lambda source, root: None)
    monkeypatch.setattr(merger, "_excluded_identities", lambda root, value: (set(), set()))
    monkeypatch.setattr(
        merger,
        "_coordinates",
        lambda root, value: np.asarray([[300.0, -60.0]], dtype=np.float64),
    )
    manifest = merger.build_sample_manifest(tmp_path)
    assert manifest["counts"]["selected"] == 400
    assert manifest["counts"]["exploration"] == 300
    assert manifest["counts"]["reserved_confirmation"] == 100
    assert manifest["counts"]["predecessor_selected"] == 0
    assert manifest["counts"]["response_rows_read"] == 0
    assert all(row["response_read"] is False for row in manifest["objects"])
    assert set(manifest["cell_counts"].values()) == {100}
    assert manifest["fold_counts_exploration"] == {str(index): 60 for index in range(5)}
    cells = Counter((row["tidal_bin"], row["stellar_mass_bin"]) for row in manifest["objects"])
    assert set(cells.values()) == {100}


def test_pseudorandom_generator_is_deterministic_and_fully_labeled() -> None:
    config = merger.load_config(ROOT)
    first = merger.generate_candidates(config)
    second = merger.generate_candidates(config)
    assert merger._candidate_digest(first) == merger._candidate_digest(second)
    assert len(first["family"]) == 262144
    assert all(np.array_equal(first[key], second[key]) for key in first)
    manifest = merger.build_candidate_manifest(ROOT)
    assert sum(manifest["family_counts"].values()) == 262144
    assert sum(manifest["origin_status_counts"].values()) == 262144
    assert manifest["counts"]["post_response_cells"] == 0
    assert manifest["counts"]["response_rows_read"] == 0
    assert manifest["claims"]["historical_novelty_established"] is False


def test_all_relaxation_families_are_finite_and_distinct() -> None:
    candidates = {
        "family": np.arange(12, dtype=np.int16),
        "threshold": np.linspace(-1.1, 1.1, 12),
        "scale": np.linspace(0.2, 1.3, 12),
        "power": np.linspace(0.5, 3.0, 12),
        "phase": np.linspace(0.1, 2.0, 12),
        "modulation": np.zeros(12, dtype=np.int8),
    }
    count = 19
    data = {
        "tidal": np.resize(np.asarray([-1.0, 1.0]), count),
        "merger": np.resize(np.asarray([1.0, -1.0, -1.0]), count),
        "asymmetry": np.linspace(-2, 2, count),
        "asymmetry_error": np.linspace(0.1, 1.1, count),
        "clumpiness": np.linspace(1.7, -1.3, count),
        "bar": np.linspace(-1.5, 1.5, count),
        "surface": np.linspace(-2, 2, count),
        "mass_modulation": np.linspace(-1, 1.5, count),
        "axis_modulation": np.linspace(-1.5, 1.5, count),
        "prior_age": np.linspace(-0.8, 0.8, count),
    }
    components = merger._candidate_components(candidates, data, 0, 12, np)
    assert components.shape == (12, count)
    assert np.all(np.isfinite(components))
    assert len({np.round(row, 10).tobytes() for row in components}) == 12


def test_nested_selector_executes_all_four_primary_models_and_secondary() -> None:
    config = copy.deepcopy(merger.load_config(ROOT))
    config["candidate_generator"]["candidate_cells"] = 96
    config["evaluation"]["candidate_batch_size"] = 24
    config["evaluation"]["cpu_crosscheck_candidates"] = 8
    count = 25
    random = np.random.default_rng(13)
    prior_age = random.normal(size=count)
    structural = random.normal(size=(count, 13))
    data = {
        "folds": np.arange(count) % 5,
        "y": 2.0 + 0.04 * prior_age + random.normal(0, 0.05, count),
        "y_span": 2.1 + random.normal(0, 0.08, count),
        "design_structural": structural,
        "design_age": np.column_stack((structural, prior_age)),
        "prior_age": prior_age,
        "tidal": np.resize(np.asarray([-1.0, 1.0]), count),
        "merger": np.resize(np.asarray([-1.0, -1.0, 1.0]), count),
        "asymmetry": random.normal(size=count),
        "asymmetry_error": np.abs(random.normal(size=count)),
        "clumpiness": random.normal(size=count),
        "bar": random.normal(size=count),
        "surface": random.normal(size=count),
        "mass_modulation": random.normal(size=count),
        "axis_modulation": random.normal(size=count),
        "mass": random.normal(size=count),
    }
    predictions, selections, compute = merger._nested_select(data, config)
    assert set(predictions) == {
        "structural",
        "age",
        "full",
        "disturbance_only",
        "span_age",
        "span_full",
    }
    assert all(np.all(np.isfinite(value)) for value in predictions.values())
    assert len(selections) == 5
    assert compute["candidate_cells"] == 96
    assert compute["candidate_galaxy_score_evaluations"] == 96 * count * 20


def test_formula_and_sample_builders_have_no_response_parameter() -> None:
    for builder in (
        merger.derive_predictors,
        merger._prior_age_component,
        merger.generate_candidates,
        merger.build_candidate_manifest,
        merger.build_sample_manifest,
    ):
        signature = " ".join(inspect.signature(builder).parameters).lower()
        assert "velocity" not in signature
        assert "response" not in signature
    component_source = inspect.getsource(merger._candidate_components)
    assert "median" not in component_source
    assert "std" not in component_source
    response_source = inspect.getsource(merger.write_response_source)
    assert 'row["role"] == "exploration"' in response_source
    assert 'row["role"] == "reserved_confirmation"' in response_source
    query = merger._response_query(merger.load_config(ROOT), ["1-1"])
    assert "d.plateifu IN" in query


def test_stored_artifacts_replay_if_present() -> None:
    config = merger.load_config(ROOT)
    validators = (
        ("predictor_source", merger.validate_predictor_source),
        ("sample_manifest", merger.validate_sample_manifest),
        ("candidate_manifest", merger.validate_candidate_manifest),
        ("response_source", merger.validate_response_source),
    )
    for key, validator in validators:
        path = ROOT / config["outputs"][key]
        if path.exists():
            validator(json.loads(path.read_text(encoding="utf-8")), ROOT)


def test_stored_result_replays_if_present() -> None:
    config = merger.load_config(ROOT)
    path = ROOT / config["outputs"]["result"]
    if not path.exists():
        pytest.skip("MaNGA relaxation exploration has not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    merger.validate_receipt(stored, ROOT)
    merger.check_receipt(ROOT)
    assert stored["counts"]["candidate_cells"] == 262144
    assert stored["counts"]["confirmation_response_rows"] == 0
    assert stored["counts"]["post_response_formula_cells"] == 0
    assert stored["counts"]["paid_model_calls"] == 0
