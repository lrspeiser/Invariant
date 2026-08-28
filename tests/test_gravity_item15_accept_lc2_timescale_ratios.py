from __future__ import annotations

import copy
import inspect
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item15_accept_lc2_timescale_ratios as cluster

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_direct_cooling_lensing_and_incident_boundaries() -> None:
    config = cluster.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 15
    assert config["predecessor"]["required_decision"] == (
        "INCONCLUSIVE_ITEM15_MANGA_TIMESCALE_QUALITY"
    )
    assert config["sample"]["eligible_objects"] == 23
    assert config["sample"]["exploration_objects"] == 18
    assert config["sample"]["confirmation_objects"] == 5
    assert config["candidate_generator"]["candidate_cells"] == 262144
    assert len(config["candidate_generator"]["families"]) == 12
    assert config["claim_boundaries"]["direct_hot_gas_cooling_time_tested"] is True
    assert config["authorization"]["lc2_response_query_allowed_before_sample_freeze"] is False
    assert config["authorization"]["confirmation_response_query_allowed"] is False
    assert config["authorization"]["post_response_candidate_generation_allowed"] is False
    incident = config["prefreeze_access"]["incident"]
    assert incident["numeric_response_rows_accidentally_displayed"] == 6
    assert incident["paper_sample_objects"] == 100
    assert len(incident["overlapping_accept_names"]) == 47


def _profile() -> list[dict[str, float]]:
    rows = []
    for inner, outer, density, temperature, cooling in (
        (0.0, 0.03, 0.04, 4.0, 0.8),
        (0.03, 0.07, 0.015, 5.0, 2.5),
        (0.07, 0.12, 0.005, 6.0, 7.0),
    ):
        row = {field: 1.0 for field in cluster.PROFILE_FIELDS}
        row.update(
            {
                "radius_inner_mpc": inner,
                "radius_outer_mpc": outer,
                "electron_density_cm3": density,
                "temperature_kev": temperature,
                "cooling_time_isochoric_gyr": cooling,
                "cooling_time_isobaric_gyr": cooling * 5.0 / 3.0,
            }
        )
        rows.append(row)
    return rows


def test_direct_physical_clocks_are_positive_dimensionless_and_mass_grows() -> None:
    config = cluster.load_config(ROOT)
    values = cluster.derive_cluster_features(
        _profile(),
        {
            "accept_name": "TEST",
            "ra": 10.0,
            "dec": 20.0,
            "redshift": 0.2,
            "k0": 20.0,
            "k100": 150.0,
            "entropy_alpha": 1.1,
            "global_temperature_kev": 5.0,
            "lbol_1e44_erg_s": 10.0,
        },
        {"lc2_name": "TEST", "lc2_author": "test", "lc2_bibcode": "test"},
        config,
    )
    assert values["mgas20_msun"] < values["mgas50_msun"] < values["mgas100_msun"]
    assert values["tff_baryon20_gyr"] > 0
    assert values["tsound100_gyr"] > 0
    assert values["cosmic_age_gyr"] > 0
    assert 0.0 < values["clock_hierarchy_entropy"] <= 1.0
    assert values["clock_hierarchy_span"] >= 0.0
    for radius in (20, 50, 100):
        assert math.isfinite(values[f"log_tcool_tff{radius}"])
        assert math.isfinite(values[f"log_tcool_tsound{radius}"])
        assert math.isfinite(values[f"log_tcool_cosmic{radius}"])


def test_candidate_generator_is_deterministic_labeled_and_balanced() -> None:
    config = cluster.load_config(ROOT)
    first = cluster.generate_candidates(config)
    second = cluster.generate_candidates(config)
    assert cluster._candidate_digest(first) == cluster._candidate_digest(second)
    assert len(first["family"]) == 262144
    assert set(np.unique(first["family"])) == set(range(12))
    assert set(np.unique(first["radius"])) == {0, 1, 2}
    families = config["candidate_generator"]["families"]
    assert sum(not family["qualifying"] for family in families) == 3
    assert all(family["origin_status"] for family in families)


def _candidate_fixture(count: int = 27) -> dict[str, object]:
    random = np.random.default_rng(15152)
    return {
        "cooling_freefall": random.normal(size=(3, count)),
        "cooling_crossing": random.normal(size=(3, count)),
        "cooling_cosmic": random.normal(size=(3, count)),
        "crossing_freefall": random.normal(size=(3, count)),
        "hierarchy_span": np.abs(random.normal(size=count)),
        "hierarchy_entropy": random.uniform(0.1, 1.0, size=count),
        "core_entropy_modulation": random.normal(size=count),
        "temperature_modulation": random.normal(size=count),
        "gas_concentration_modulation": random.normal(size=count),
        "redshift_modulation": random.normal(size=count),
    }


def test_candidate_families_are_finite_and_empirically_distinct() -> None:
    data = _candidate_fixture()
    arrays = {
        "family": np.arange(12, dtype=np.int8),
        "radius": np.arange(12, dtype=np.int8) % 3,
        "threshold": np.linspace(-1.5, 1.5, 12),
        "scale": np.linspace(0.2, 1.3, 12),
        "power": np.linspace(0.5, 3.0, 12),
        "phase": np.linspace(0.1, 5.5, 12),
        "modulation": np.zeros(12, dtype=np.int8),
    }
    components = cluster._candidate_components(arrays, data, 0, 12, np)
    assert components.shape == (12, 27)
    assert np.all(np.isfinite(components))
    assert len({np.round(row, 10).tobytes() for row in components}) == 12


def test_nested_selector_runs_small_target_blind_candidate_bank() -> None:
    config = copy.deepcopy(cluster.load_config(ROOT))
    config["candidate_generator"]["candidate_cells"] = 120
    config["evaluation"]["candidate_batch_size"] = 30
    config["evaluation"]["cpu_crosscheck_candidates"] = 12
    arrays = cluster.generate_candidates(config)
    count = 25
    random = np.random.default_rng(15153)
    data = {
        **_candidate_fixture(count),
        "folds": np.arange(count) % 5,
        "design": random.normal(size=(count, 9)),
    }
    components, crosscheck = cluster._component_matrix(arrays, data, config, np)
    y = random.normal(size=count)
    baseline, full, records = cluster._nested_select_matrix(
        y, data, arrays, components, config, np, include_records=True
    )
    assert np.all(np.isfinite(baseline))
    assert np.all(np.isfinite(full))
    assert len(records) == 5
    assert crosscheck == pytest.approx(0.0)


def test_lc2_parser_requires_exact_frozen_source_row() -> None:
    payload = (
        b"Name\t_RAJ2000\t_DEJ2000\tz\tNameNED\tAuthor\tBibCode\tM500\te_M500\n"
        b"TEST CLUSTER\t10.0\t20.0\t0.2\tTEST\ttest+20\t2020TEST\t5.0\t1.0\n"
    )
    row = cluster.parse_lc2_response(
        payload,
        {
            "accept_name": "TEST_CLUSTER",
            "lc2_name": "TEST CLUSTER",
            "lc2_author": "test+20",
            "lc2_bibcode": "2020TEST",
            "ra": 10.0,
            "dec": 20.0,
            "redshift": 0.2,
        },
    )
    assert row["m500_1e14_msun"] == 5.0
    assert row["m500_error_1e14_msun"] == 1.0


def test_formula_and_sample_builders_have_no_lensing_response_input() -> None:
    for builder in (
        cluster.derive_cluster_features,
        cluster.generate_candidates,
        cluster.write_prepared_sources,
    ):
        signature = " ".join(inspect.signature(builder).parameters).lower()
        assert "m500" not in signature
        assert "response" not in signature
    preparation_source = inspect.getsource(cluster.write_prepared_sources).lower()
    assert "parse_lc2_response" not in preparation_source
    assert "m500" not in preparation_source
    response_source = inspect.getsource(cluster.write_response_source)
    assert 'row["role"] == "exploration"' in response_source
    assert 'row["role"] == "reserved_confirmation"' in response_source


def test_access_requires_exact_commit_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cluster, "SCIENTIFIC_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(cluster.GravityItem15AcceptLC2Error, match="not bound"):
        cluster.write_prepared_sources(ROOT)
    monkeypatch.setattr(cluster, "SAMPLE_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(cluster.GravityItem15AcceptLC2Error, match="not bound"):
        cluster.write_response_source(ROOT)


def test_stored_artifacts_replay_if_present() -> None:
    config = cluster.load_config(ROOT)
    prepared_paths = [
        ROOT / config["outputs"][key]
        for key in ("sample_manifest", "predictor_source", "candidate_manifest")
    ]
    if all(path.exists() for path in prepared_paths):
        values = [json.loads(path.read_text(encoding="utf-8")) for path in prepared_paths]
        cluster.validate_prepared_sources(*values, ROOT)
    response_path = ROOT / config["outputs"]["response_source"]
    if response_path.exists():
        cluster.validate_response_source(
            json.loads(response_path.read_text(encoding="utf-8")), ROOT
        )


def test_stored_result_replays_if_present() -> None:
    config = cluster.load_config(ROOT)
    path = ROOT / config["outputs"]["result"]
    if not path.exists():
        pytest.skip("Item 15 ACCEPT/LC2 exploration has not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    cluster.validate_receipt(stored, ROOT)
    cluster.check_receipt(ROOT)
    assert stored["counts"]["confirmation_response_rows"] == 0
    assert stored["counts"]["post_response_formula_cells"] == 0
    assert stored["counts"]["paid_model_calls"] == 0
    assert Counter(
        row["outer_fold"]
        for row in stored["primary_lensing_to_gas_log_mass_ratio"]["outer_fold_selections"]
    ) == Counter(range(5))
