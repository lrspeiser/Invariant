from __future__ import annotations

import copy
import inspect
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item15_manga_timescale_ratios as timescale

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_scope_equivalence_and_response_boundaries() -> None:
    config = timescale.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 15
    assert config["predecessor"]["required_decision"] == (
        "REJECT_ITEM14_MASK_COHERENCE_ADVANCE_ITEM15"
    )
    assert config["sample"]["maximum_total_objects"] == 320
    assert config["sample"]["exploration_objects"] == 240
    assert config["sample"]["confirmation_objects"] == 80
    assert config["sample"]["prefreeze_predictor_audit"]["eligible"] == 2366
    assert min(
        config["sample"]["prefreeze_predictor_audit"]["cell_counts"].values()
    ) == 113
    assert config["candidate_generator"]["candidate_cells"] == 262144
    assert len(config["candidate_generator"]["families"]) == 12
    assert config["timescale_features"]["direct_hot_gas_cooling_time_available"] is False
    assert config["timescale_features"]["direct_hot_gas_cooling_followup_required"] is True
    assert config["authorization"]["response_query_allowed_before_sample_freeze"] is False
    assert config["authorization"]["confirmation_response_query_allowed"] is False
    assert config["authorization"]["post_response_candidate_generation_allowed"] is False
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert all(value is False for value in config["claim_boundaries"].values())


def _predictor(log_specific_sfr: float = -11.0) -> dict[str, float]:
    return {
        "log_stellar_mass": 10.0,
        "log_half_light_radius": math.log10(5.0),
        "log_specific_sfr": log_specific_sfr,
        "redshift": 0.03,
    }


def test_physical_timescales_preserve_known_equivalence_class() -> None:
    config = timescale.load_config(ROOT)
    values = timescale.derive_timescale_features(_predictor(), config)
    dynamical = values["dynamical_time_gyr"]
    assert values["crossing_time_gyr"] == pytest.approx(dynamical)
    assert values["orbital_time_gyr"] / dynamical == pytest.approx(2.0 * math.pi)
    assert values["free_fall_time_gyr"] / dynamical == pytest.approx(
        math.pi / math.sqrt(8.0)
    )
    assert 0.0 < values["clock_entropy"] <= 1.0
    assert values["clock_hierarchy_span"] >= 0.0


def test_mass_doubling_and_cosmic_clocks_move_in_physical_directions() -> None:
    config = timescale.load_config(ROOT)
    active = timescale.derive_timescale_features(_predictor(-10.0), config)
    quiet = timescale.derive_timescale_features(_predictor(-12.0), config)
    assert quiet["mass_doubling_time_gyr"] / active["mass_doubling_time_gyr"] == pytest.approx(
        100.0
    )
    assert timescale.cosmic_age_gyr(0.0, config) > timescale.cosmic_age_gyr(0.1, config)


def test_response_free_fresh_predictor_audit_replays() -> None:
    config = timescale.load_config(ROOT)
    rows, failures = timescale._eligible_predictors(ROOT, config)
    assert len(rows) == 2366
    assert failures == config["sample"]["prefreeze_predictor_audit"]["failure_counts"]
    assert Counter(row["sample_cell"] for row in rows) == Counter(
        config["sample"]["prefreeze_predictor_audit"]["cell_counts"]
    )


def test_candidate_generator_is_deterministic_and_provenance_labeled() -> None:
    config = timescale.load_config(ROOT)
    first = timescale.generate_candidates(config)
    second = timescale.generate_candidates(config)
    assert timescale._candidate_digest(first) == timescale._candidate_digest(second)
    assert len(first["family"]) == 262144
    assert set(np.unique(first["family"])) == set(range(12))
    assert all(family["origin_status"] for family in config["candidate_generator"]["families"])
    assert sum(not family["qualifying"] for family in config["candidate_generator"]["families"]) == 1


def test_candidate_families_are_finite_and_empirically_distinct() -> None:
    count = 31
    random = np.random.default_rng(1515)
    arrays = {
        "family": np.arange(12, dtype=np.int8),
        "threshold": np.linspace(-1.0, 2.0, 12),
        "scale": np.linspace(0.2, 1.3, 12),
        "power": np.linspace(0.5, 3.0, 12),
        "phase": np.linspace(0.1, 5.5, 12),
        "modulation": np.zeros(12, dtype=np.int8),
    }
    data = {
        key: random.normal(size=count)
        for key in (
            "cosmic_dynamical",
            "growth_dynamical",
            "growth_cosmic",
            "relaxation_cosmic",
            "relaxation_growth",
            "clock_hierarchy",
            "clock_entropy",
            "surface_modulation",
            "age_modulation",
            "sersic_modulation",
            "mass_modulation",
            "redshift_modulation",
        )
    }
    components = timescale._candidate_components(arrays, data, 0, 12, np)
    assert components.shape == (12, count)
    assert np.all(np.isfinite(components))
    assert len({np.round(row, 10).tobytes() for row in components}) == 12


def test_nested_selector_executes_primary_and_honest_secondary() -> None:
    config = copy.deepcopy(timescale.load_config(ROOT))
    config["candidate_generator"]["candidate_cells"] = 96
    config["evaluation"]["candidate_batch_size"] = 24
    config["evaluation"]["cpu_crosscheck_candidates"] = 8
    count = 25
    random = np.random.default_rng(1516)
    data = {
        "folds": np.arange(count) % 5,
        "y": random.normal(0.1, 0.15, count),
        "y_halpha": random.normal(0.08, 0.15, count),
        "design_control": random.normal(size=(count, 16)),
        "design_secondary": random.normal(size=(count, 16)),
    }
    for key in (
        "cosmic_dynamical",
        "growth_dynamical",
        "growth_cosmic",
        "relaxation_cosmic",
        "relaxation_growth",
        "clock_hierarchy",
        "clock_entropy",
        "surface_modulation",
        "age_modulation",
        "sersic_modulation",
        "mass_modulation",
        "redshift_modulation",
    ):
        data[key] = random.normal(size=count)
    predictions, selections, compute = timescale._nested_select(data, config)
    assert set(predictions) == {"control", "full", "secondary_control", "secondary_full"}
    assert all(np.all(np.isfinite(value)) for value in predictions.values())
    assert len(selections) == 5
    assert compute["candidate_cells"] == 96
    assert compute["candidate_galaxy_score_evaluations"] == 96 * count * 20


def test_formula_and_sample_builders_have_no_response_input() -> None:
    for builder in (
        timescale.derive_timescale_features,
        timescale.generate_candidates,
        timescale._eligible_predictors,
        timescale.write_prepared_sources,
    ):
        signature = " ".join(inspect.signature(builder).parameters).lower()
        assert "velocity" not in signature
        assert "response" not in signature
    preparation_source = inspect.getsource(timescale.write_prepared_sources)
    assert "_maps_payload" not in preparation_source
    assert "derive_radial_response" not in preparation_source
    response_source = inspect.getsource(timescale.write_response_source)
    assert 'row["role"] == "exploration"' in response_source
    assert 'row["role"] == "reserved_confirmation"' in response_source


def test_access_requires_exact_commit_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timescale, "SCIENTIFIC_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(timescale.GravityItem15TimescaleError, match="not bound"):
        timescale.write_prepared_sources(ROOT)
    monkeypatch.setattr(timescale, "SAMPLE_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(timescale.GravityItem15TimescaleError, match="not bound"):
        timescale.write_response_source(ROOT)


def test_stored_artifacts_replay_if_present() -> None:
    config = timescale.load_config(ROOT)
    prepared_paths = [
        ROOT / config["outputs"][key]
        for key in ("sample_manifest", "predictor_source", "candidate_manifest")
    ]
    if all(path.exists() for path in prepared_paths):
        values = [json.loads(path.read_text(encoding="utf-8")) for path in prepared_paths]
        timescale.validate_prepared_sources(*values, ROOT)
    response_path = ROOT / config["outputs"]["response_source"]
    if response_path.exists():
        timescale.validate_response_source(
            json.loads(response_path.read_text(encoding="utf-8")), ROOT
        )


def test_stored_result_replays_if_present() -> None:
    config = timescale.load_config(ROOT)
    path = ROOT / config["outputs"]["result"]
    if not path.exists():
        pytest.skip("Item 15 galaxy timescale exploration has not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    timescale.validate_receipt(stored, ROOT)
    timescale.check_receipt(ROOT)
    assert stored["counts"]["confirmation_response_rows"] == 0
    assert stored["counts"]["post_response_formula_cells"] == 0
    assert stored["counts"]["paid_model_calls"] == 0
