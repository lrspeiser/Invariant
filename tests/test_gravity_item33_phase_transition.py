from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import _read_json, _read_tsv
from sigma_theory_compiler.gravity_item33_phase_transition import (
    _admissible_candidates,
    _candidate_activation,
    _candidate_delta_log10_sigma,
    _candidate_manifest,
    _candidate_record,
    _candidate_values,
    _cell_residual_permutation,
    _contract_digest,
    _fresh_pool,
    _ordinary_change_point_design,
    _parse_skyserver_csv,
    _response_query,
    _sample_manifest,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item33_config_preserves_strict_boundary() -> None:
    config = load_config(ROOT)
    assert config["item"] == 33
    assert config["stable_goal_sha256"] == (
        "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
    )
    assert config["discovery_policy"]["equal_initial_viability"] is True
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["scope"]["paid_api_calls_authorized"] is False
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert "per_object_critical_threshold" in config["scope"]["forbidden_inputs"]
    assert config["sources"]["inherited_predictor_response_columns_read"] == 0
    assert "change-point" in config["evaluation"]["permutation_strategy"]


def test_item33_contract_digest_ignores_only_bound_commit_ids() -> None:
    config = load_config(ROOT)
    changed = json.loads(json.dumps(config))
    changed["scientific_freeze_commit"] = "a" * 40
    changed["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["gates"]["maximum_selection_aware_permutation_p"] = 0.5
    assert _contract_digest(changed) != _contract_digest(config)


def test_item33_raw_grammar_has_exact_equal_unique_niches() -> None:
    arrays = generate_raw_candidates(load_config(ROOT))
    assert len(arrays["niche"]) == 262144
    assert Counter(arrays["niche"].tolist()) == {0: 65536, 1: 65536, 2: 65536, 3: 65536}
    signatures = np.column_stack([arrays[key] for key in sorted(arrays)])
    assert len(np.unique(signatures, axis=0)) == 262144
    for niche in range(4):
        mask = arrays["niche"] == niche
        assert np.count_nonzero(arrays["polarity"][mask] == 0) == 32768
        assert np.count_nonzero(arrays["polarity"][mask] == 1) == 32768


def test_item33_admissibility_is_frozen_local_bounded_and_converged() -> None:
    config = load_config(ROOT)
    arrays, audit = _admissible_candidates(config)
    generator = config["candidate_generator"]
    assert audit["raw_candidate_digest"] == generator["expected_raw_candidate_digest"]
    assert audit["admissible_candidate_digest"] == generator["expected_admissible_candidate_digest"]
    assert audit["admissible_candidates"] == generator["expected_admissible_candidates"]
    assert audit["admissible_per_niche"] == generator["expected_admissible_per_niche"]
    assert (
        audit["behavioral_equivalence_classes_adversarial"]
        == generator["expected_behavioral_equivalence_classes_adversarial"]
    )
    assert set(arrays["niche"].tolist()) == {0, 1, 2, 3}
    assert audit["minimum_admitted_mu"] >= config["admissibility"]["minimum_mu"]
    assert audit["maximum_admitted_mu"] <= config["admissibility"]["maximum_mu"]
    assert (
        audit["minimum_admitted_phase_activation_span"]
        >= config["admissibility"]["minimum_phase_activation_span"]
    )
    assert (
        audit["maximum_admitted_local_fractional_response"]
        <= config["admissibility"]["maximum_local_fractional_response"]
    )
    assert (
        audit["maximum_admitted_landau_fixed_point_difference"]
        <= config["admissibility"]["landau_fixed_point_tolerance"]
    )


def test_item33_all_equations_are_finite_bounded_and_target_blind() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    indices = [int(np.where(arrays["niche"] == niche)[0][0]) for niche in range(4)]
    subset = {key: value[indices] for key, value in arrays.items()}
    predictors = {
        "log_acceleration": np.asarray([-13.0, -10.5, -8.5]),
        "log_density": np.asarray([5.0, 8.0, 10.5]),
        "log_potential": np.asarray([-8.5, -6.5, -4.5]),
        "q_lss": np.asarray([-5.0, -2.0, 2.0]),
        "age": np.asarray([-2.0, 0.0, 2.0]),
        "log_surface_density": np.asarray([6.5, 8.8, 11.0]),
        "axis_ratio": np.asarray([0.25, 0.6, 0.95]),
    }
    activation = _candidate_activation(config, subset, predictors, 0, 4, np)
    delta = _candidate_delta_log10_sigma(config, subset, predictors, 0, 4, np)
    assert activation.shape == (4, 3)
    assert np.all(np.isfinite(activation))
    assert np.all((activation >= 0.0) & (activation <= 1.0))
    assert np.all(np.isfinite(delta))
    assert "target" not in _candidate_activation.__annotations__


def test_item33_labels_keep_new_synthesis_without_claiming_novelty() -> None:
    manifest = _candidate_manifest(load_config(ROOT))
    assert manifest["historical_novelty_claimed"] is False
    assert manifest["post_response_cells"] == 0
    labels = {row["creativity_label"] for row in manifest["niches"]}
    assert labels == {
        "known_family_control",
        "known_family_extension",
        "known_family_combination",
        "potentially_new_synthesis",
    }
    assert manifest["audit"]["raw_per_niche"] == {
        "0": 65536,
        "1": 65536,
        "2": 65536,
        "3": 65536,
    }


def test_item33_real_response_blind_pool_excludes_all_predecessor_roles() -> None:
    config = load_config(ROOT)
    pool, prior_ids = _fresh_pool(ROOT, config)
    inherited = _read_tsv(ROOT / config["sources"]["inherited_predictors"])
    union = set()
    for key in ("item30_sample_manifest", "item31_sample_manifest", "item32_sample_manifest"):
        predecessor = _read_json(ROOT / config["sources"][key])
        union.update(str(row["plateifu"]) for row in predecessor["objects"])
    assert prior_ids == union
    assert len(prior_ids) == 1248
    assert len(inherited) == 1511
    assert len(pool) == 189
    assert not ({str(row["plateifu"]) for row in pool} & prior_ids)
    sample = _sample_manifest(config, pool)
    assert sample["counts"] == {
        "fresh_predictor_pool": 189,
        "selected": 180,
        "exploration": 160,
        "reserved_confirmation": 20,
        "response_rows_read": 0,
    }
    assert sample["fold_counts_exploration"] == {str(fold): 32 for fold in range(5)}
    assert {
        key: value["eligible"] for key, value in sample["selected_cell_counts"].items()
    } == config["sample"]["expected_cell_capacities"]
    assert all(row["response_read"] is False for row in sample["objects"])


def test_item33_response_query_contains_only_declared_ids_and_columns() -> None:
    config = load_config(ROOT)
    query = _response_query(config, ["1000-1901", "1001-1902"])
    for column in config["sources"]["response_columns"]:
        assert f"d.{column}" in query
    assert "1000-1901" in query and "1001-1902" in query
    assert "reserved_confirmation" not in query
    assert "dark" not in query.lower()


def test_item33_skyserver_parser_handles_declared_comment_metadata() -> None:
    payload = (
        b"#Table1\nplateifu,stellar_sigma_1re,stellar_rchi2_1re,"
        b"stellar_vel_lo_clip,stellar_vel_hi_clip\n"
        b"1000-1901,120,1.2,-90,95\n"
    )
    rows, comments = _parse_skyserver_csv(payload)
    assert comments == ["#Table1"]
    assert rows[0]["plateifu"] == "1000-1901"
    assert rows[0]["stellar_sigma_1re"] == "120"


def test_item33_change_point_control_is_response_blind_and_deterministic() -> None:
    config = load_config(ROOT)
    pool, _ = _fresh_pool(ROOT, config)
    design_a, thresholds_a = _ordinary_change_point_design(pool, config)
    design_b, thresholds_b = _ordinary_change_point_design(pool, config)
    assert design_a.shape == design_b.shape
    assert design_a.shape[0] == 189
    assert design_a.shape[1] > 126
    assert np.array_equal(design_a, design_b)
    assert thresholds_a == thresholds_b
    assert set(thresholds_a) == {
        "log_acceleration",
        "log_density",
        "log_potential",
        "q_lss",
        "log_surface_density",
        "age",
        "axis_ratio",
    }
    assert "target" not in _ordinary_change_point_design.__annotations__


def test_item33_residual_null_preserves_each_response_blind_cell() -> None:
    target = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    reference = np.asarray([0.5, 0.5, 0.5, 5.0, 5.0, 5.0])
    rows = [
        {"sample_cell": "g0-env0"},
        {"sample_cell": "g0-env0"},
        {"sample_cell": "g0-env0"},
        {"sample_cell": "g1-env1"},
        {"sample_cell": "g1-env1"},
        {"sample_cell": "g1-env1"},
    ]
    null = _cell_residual_permutation(
        target, reference, rows, np.random.Generator(np.random.PCG64(330699))
    )
    for indices in (np.asarray([0, 1, 2]), np.asarray([3, 4, 5])):
        assert sorted((null - reference)[indices]) == sorted((target - reference)[indices])


def test_item33_candidate_decoder_reports_critical_parameters() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    values = _candidate_values(config, arrays, 0, 3, np)
    assert values["polarity"].shape == (3,)
    assert np.all(np.isin(values["polarity"], [-1.0, 1.0]))
    assert np.all(values["amplitude"] > 0.0)
    assert np.all(np.isfinite(values["acceleration_threshold"]))
    for niche in range(4):
        index = int(np.where(arrays["niche"] == niche)[0][0])
        record = _candidate_record(index, config, arrays)
        assert record["niche_index"] == niche
        assert record["creativity_label"]
