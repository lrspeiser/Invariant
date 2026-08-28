from __future__ import annotations

import copy
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item12_manga_dynamical_age as age

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_fresh_real_source_generator_and_response_boundary() -> None:
    config = age.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 12
    assert config["predecessor"]["required_decision"] == (
        "INCONCLUSIVE_ITEM11_QUALITY_NEGATIVE_DIRECTION_ADVANCE_ITEM12"
    )
    assert config["source"]["observed_join_rows"] == 10782
    assert config["source"]["prefreeze_queries"]["predictor_rows_read"] == 0
    assert config["source"]["prefreeze_queries"]["response_rows_read"] == 0
    assert config["candidate_generator"]["candidate_cells"] == 262144
    assert len(config["candidate_generator"]["families"]) == 12
    assert config["candidate_generator"]["historical_novelty_claimed"] is False
    assert config["source"]["confirmation_query_forbidden"] is True
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["authorization"]["stellar_kinematic_response_as_predictor_allowed"] is False
    assert config["quality"]["require_dapqual_zero"] is True
    assert config["quality"]["require_drp3qual_zero"] is True
    assert "no held-out predictor or response" in config["evaluation"]["normalization_rule"]


def _synthetic_predictor() -> dict[str, str]:
    return {
        "plateifu": "1000-12701",
        "mangaid": "1-100",
        "objra": "150",
        "objdec": "2",
        "daptype": "HYB10-MILESHC-MASTARSSP",
        "dapqual": "0",
        "snr_med_g": "20",
        "sb_1re": "5",
        "specindex_1re_dn4000": "1.6",
        "specindex_1re_d4000": "1.7",
        "specindex_1re_hdeltaa": "3",
        "specindex_1re_hgammaa": "1",
        "specindex_1re_hb": "2",
        "emline_sew_1re_ha_6564": "-10",
        "sfr_1re": "0.3",
        "sfr_tot": "1.0",
        "nsa_elpetro_mass": "10000000000",
        "nsa_elpetro_th50_r": "5",
        "nsa_elpetro_ba": "0.7",
        "nsa_sersic_n": "2",
        "nsa_sersic_absmag_g": "-20",
        "nsa_sersic_absmag_r": "-21",
        "nsa_z": "0.03",
        "drp3qual": "0",
    }


def test_skyserver_parser_query_scope_and_predictor_derivation() -> None:
    assert age._parse_skyserver_csv(b"# comment\na,b\n 1, hello \n") == [{"a": "1", "b": "hello"}]
    config = age.load_config(ROOT)
    query = age._predictor_query(config)
    assert "stellar_sigma_1re" not in query
    assert "stellar_vel" not in query
    assert config["source"]["daptype"] in query
    derived = age.derive_predictors(_synthetic_predictor(), config)
    assert derived["log_stellar_mass"] == pytest.approx(10)
    assert derived["g_minus_r_color"] == pytest.approx(1)
    assert derived["log_specific_sfr"] < -10
    assert derived["mass_size_crossing_proxy"] < 0


def test_predictor_quality_bitmasks_are_target_blind_exclusions() -> None:
    config = age.load_config(ROOT)
    for field in ("dapqual", "drp3qual"):
        row = _synthetic_predictor()
        row[field] = "1"
        with pytest.raises(age.GravityItem12DynamicalAgeError, match="physical range"):
            age.derive_predictors(row, config)


def test_access_requires_exact_commit_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(age, "SCIENTIFIC_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(age.GravityItem12DynamicalAgeError, match="not bound"):
        age.write_predictor_source(ROOT)
    monkeypatch.setattr(age, "SAMPLE_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(age.GravityItem12DynamicalAgeError, match="not bound"):
        age.write_response_source(ROOT)


def test_predecessor_registry_includes_records_format_and_prior_manga() -> None:
    config = age.load_config(ROOT)
    prior = age._prior_manga_ids(ROOT, config)
    coordinates = age._coordinates(ROOT, config)
    assert len(prior) == 90
    assert coordinates.shape[0] > 1000
    assert coordinates.shape[1] == 2
    assert any(value.startswith("1-") for value in prior)


def test_target_blind_manifest_balances_four_cells_and_seals_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(age.load_config(ROOT))
    config["outputs"]["predictor_source"] = "predictor-source.json"
    records = []
    ordinal = 0
    for mass, dn4000 in ((9.0, 1.0), (9.0, 2.0), (11.0, 1.0), (11.0, 2.0)):
        for _ in range(300):
            ordinal += 1
            records.append(
                {
                    "plateifu": f"{ordinal}-1",
                    "mangaid": f"1-{ordinal}",
                    "ra": "10",
                    "dec": "10",
                    "log_stellar_mass": age._metric(mass),
                    "dn4000": age._metric(dn4000),
                }
            )
    source = age._content_hashed({"records": records})
    (tmp_path / "predictor-source.json").write_bytes(age.canonical_json_bytes(source) + b"\n")
    monkeypatch.setattr(age, "load_config", lambda root: config)
    monkeypatch.setattr(age, "validate_predictor_source", lambda source, root: None)
    monkeypatch.setattr(age, "_prior_manga_ids", lambda root, value: set())
    monkeypatch.setattr(
        age,
        "_coordinates",
        lambda root, value: np.asarray([[300.0, -60.0]], dtype=np.float64),
    )
    manifest = age.build_sample_manifest(tmp_path)
    assert manifest["counts"]["selected"] == 1000
    assert manifest["counts"]["exploration"] == 750
    assert manifest["counts"]["reserved_confirmation"] == 250
    assert manifest["counts"]["predecessor_selected"] == 0
    assert manifest["counts"]["response_rows_read"] == 0
    assert all(row["response_read"] is False for row in manifest["objects"])
    cells = Counter((row["age_bin"], row["stellar_mass_bin"]) for row in manifest["objects"])
    assert set(cells.values()) == {250}


def test_pseudorandom_generator_is_deterministic_and_fully_labeled() -> None:
    config = age.load_config(ROOT)
    first = age.generate_candidates(config)
    second = age.generate_candidates(config)
    assert age._candidate_digest(first) == age._candidate_digest(second)
    assert len(first["family"]) == 262144
    assert all(np.array_equal(first[key], second[key]) for key in first)
    manifest = age.build_candidate_manifest(ROOT)
    assert sum(manifest["family_counts"].values()) == 262144
    assert sum(manifest["origin_status_counts"].values()) == 262144
    assert manifest["counts"]["post_response_cells"] == 0
    assert manifest["counts"]["response_rows_read"] == 0
    assert manifest["claims"]["historical_novelty_established"] is False


def test_all_dynamical_age_families_are_finite_and_distinct() -> None:
    candidates = {
        "family": np.arange(12, dtype=np.int16),
        "threshold": np.linspace(-1.1, 1.1, 12),
        "scale": np.linspace(0.2, 1.3, 12),
        "power": np.linspace(0.5, 3.0, 12),
        "phase": np.linspace(0.1, 2.0, 12),
        "modulation": np.zeros(12, dtype=np.int8),
    }
    count = 17
    data = {
        "dn4000": np.linspace(-2, 2, count),
        "d4000": np.linspace(-1.5, 2.5, count),
        "balmer": np.linspace(2.2, -1.8, count),
        "hbeta": np.linspace(-1, 1.5, count),
        "haew": np.linspace(-2.5, 2, count),
        "ssfr": np.linspace(-2, 2, count),
        "crossing": np.linspace(-1.5, 1.5, count),
        "surface": np.linspace(-2, 2, count),
        "sersic": np.linspace(-1, 2, count),
        "axis": np.linspace(-1.5, 1.5, count),
    }
    components = age._candidate_components(candidates, data, 0, 12, np)
    assert components.shape == (12, count)
    assert np.all(np.isfinite(components))
    assert len({np.round(row, 10).tobytes() for row in components}) == 12


def test_nested_selector_executes_frozen_fold_structure_on_synthetic_data() -> None:
    config = copy.deepcopy(age.load_config(ROOT))
    config["candidate_generator"]["candidate_cells"] = 96
    config["evaluation"]["candidate_batch_size"] = 24
    config["evaluation"]["cpu_crosscheck_candidates"] = 8
    count = 25
    random = np.random.default_rng(12)
    data = {
        "folds": np.arange(count) % 5,
        "y": 2.0 + random.normal(0, 0.05, count),
        "design": random.normal(size=(count, 9)),
        "dn4000": np.linspace(-2, 2, count),
        "d4000": random.normal(size=count),
        "balmer": random.normal(size=count),
        "hbeta": random.normal(size=count),
        "haew": random.normal(size=count),
        "ssfr": random.normal(size=count),
        "crossing": random.normal(size=count),
        "surface": random.normal(size=count),
        "sersic": random.normal(size=count),
        "axis": random.normal(size=count),
    }
    baseline, proposed, selections, compute = age._nested_select(data, config)
    assert np.all(np.isfinite(baseline))
    assert np.all(np.isfinite(proposed))
    assert len(selections) == 5
    assert compute["candidate_cells"] == 96
    assert compute["candidate_galaxy_score_evaluations"] == 96 * count * 20


def test_formula_and_sample_builders_have_no_response_parameter() -> None:
    for builder in (
        age.derive_predictors,
        age.generate_candidates,
        age.build_candidate_manifest,
        age.build_sample_manifest,
    ):
        signature = " ".join(inspect.signature(builder).parameters).lower()
        assert "velocity" not in signature
        assert "response" not in signature
    component_source = inspect.getsource(age._candidate_components)
    assert "median" not in component_source
    assert "std" not in component_source
    response_source = inspect.getsource(age.write_response_source)
    assert 'row["role"] == "exploration"' in response_source
    assert 'row["role"] == "reserved_confirmation"' in response_source
    assert "d.plateifu IN" in age._response_query(age.load_config(ROOT), ["1-1"])


def test_stored_artifacts_replay_if_present() -> None:
    config = age.load_config(ROOT)
    validators = (
        ("predictor_source", age.validate_predictor_source),
        ("sample_manifest", age.validate_sample_manifest),
        ("candidate_manifest", age.validate_candidate_manifest),
        ("response_source", age.validate_response_source),
    )
    for key, validator in validators:
        path = ROOT / config["outputs"][key]
        if path.exists():
            validator(json.loads(path.read_text(encoding="utf-8")), ROOT)


def test_stored_result_replays_if_present() -> None:
    config = age.load_config(ROOT)
    path = ROOT / config["outputs"]["result"]
    if not path.exists():
        pytest.skip("MaNGA dynamical-age exploration has not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    age.validate_receipt(stored, ROOT)
    age.check_receipt(ROOT)
    assert stored["counts"]["candidate_cells"] == 262144
    assert stored["counts"]["confirmation_response_rows"] == 0
    assert stored["counts"]["post_response_formula_cells"] == 0
    assert stored["counts"]["paid_model_calls"] == 0
