from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item34_condensate_superfluid import (
    _admissible_candidates,
    _candidate_activation,
    _candidate_delta_log10_velocity,
    _candidate_manifest,
    _candidate_record,
    _cell_residual_permutation,
    _coherence_control_spec,
    _contract_digest,
    _fresh_pool,
    _ordinary_coherence_design,
    _parse_skyserver_csv,
    _response_query,
    _robust_comparison,
    _sample_manifest,
    _transfer_selected_candidates,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item34_config_preserves_strict_two_track_ontology_boundary() -> None:
    config = load_config(ROOT)
    assert config["item"] == 34
    assert config["stable_goal_sha256"] == (
        "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
    )
    assert config["discovery_policy"]["equal_initial_viability"] is True
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["scope"]["paid_api_calls_authorized"] is False
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["sources"]["inherited_predictor_response_columns_read"] == 0
    assert "per_object_boson_mass" in config["scope"]["forbidden_inputs"]
    niches = config["candidate_generator"]["niches"]
    assert Counter(row["ontology"] for row in niches) == {
        "hidden_matter_required": 2,
        "baryon_sourced_gravitational_sector": 2,
    }
    assert sum(bool(row["gravity_track_eligible"]) for row in niches) == 2
    assert config["gates"]["single_empirical_counterexample_is_veto"] is False


def test_item34_contract_digest_ignores_only_bound_commit_ids() -> None:
    config = load_config(ROOT)
    changed = json.loads(json.dumps(config))
    changed["scientific_freeze_commit"] = "a" * 40
    changed["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["gates"]["phenomenon_minimum_Halpha_transfer_improvement"] = 0.5
    assert _contract_digest(changed) != _contract_digest(config)


def test_item34_raw_grammar_has_equal_unique_niches() -> None:
    arrays = generate_raw_candidates(load_config(ROOT))
    assert len(arrays["niche"]) == 262144
    assert Counter(arrays["niche"].tolist()) == {0: 65536, 1: 65536, 2: 65536, 3: 65536}
    signatures = np.column_stack([arrays[key] for key in sorted(arrays)])
    assert len(np.unique(signatures, axis=0)) == 262144
    for niche in range(4):
        mask = arrays["niche"] == niche
        assert np.count_nonzero(arrays["polarity"][mask] == 0) == 32768
        assert np.count_nonzero(arrays["polarity"][mask] == 1) == 32768


def test_item34_admissibility_is_frozen_local_bounded_and_behaviorally_counted() -> None:
    config = load_config(ROOT)
    arrays, audit = _admissible_candidates(config)
    generator = config["candidate_generator"]
    assert audit["raw_candidate_digest"] == generator["expected_raw_candidate_digest"]
    assert audit["admissible_candidate_digest"] == generator[
        "expected_admissible_candidate_digest"
    ]
    assert audit["admissible_candidates"] == generator["expected_admissible_candidates"]
    assert audit["admissible_per_niche"] == generator["expected_admissible_per_niche"]
    assert audit["behavioral_equivalence_classes_adversarial"] == generator[
        "expected_behavioral_equivalence_classes_adversarial"
    ]
    assert set(arrays["niche"].tolist()) == {0, 1, 2, 3}
    assert audit["minimum_admitted_mu"] >= config["admissibility"]["minimum_mu"]
    assert audit["maximum_admitted_mu"] <= config["admissibility"]["maximum_mu"]
    assert audit["minimum_admitted_activation_span"] >= config["admissibility"][
        "minimum_activation_span"
    ]
    assert audit["maximum_admitted_local_fractional_response"] <= config["admissibility"][
        "maximum_local_fractional_response"
    ]


def test_item34_equations_are_finite_bounded_and_target_blind() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    indices = [int(np.where(arrays["niche"] == niche)[0][0]) for niche in range(4)]
    subset = {key: value[indices] for key, value in arrays.items()}
    predictors = {
        "log_acceleration": np.asarray([-13.0, -10.5, -8.5]),
        "log_density": np.asarray([5.0, 8.0, 10.5]),
        "log_radius": np.asarray([-0.5, 0.5, 1.3]),
        "log_mass": np.asarray([8.0, 10.0, 11.5]),
        "log_surface_density": np.asarray([6.5, 8.8, 11.0]),
        "log_vbar": np.asarray([1.0, 2.0, 2.7]),
        "phase_space": np.asarray([4.0, 8.0, 11.0]),
        "age": np.asarray([-2.0, 0.0, 2.0]),
        "sfr": np.asarray([2.0, 0.0, -2.0]),
        "axis_ratio": np.asarray([0.25, 0.6, 0.95]),
    }
    activation = _candidate_activation(config, subset, predictors, 0, 4, np)
    delta = _candidate_delta_log10_velocity(config, subset, predictors, 0, 4, np)
    assert activation.shape == (4, 3)
    assert np.all(np.isfinite(activation))
    assert np.all((activation >= 0.0) & (activation <= 1.0))
    assert np.all(np.isfinite(delta))
    assert "target" not in _candidate_activation.__annotations__


def test_item34_manifest_preserves_hidden_matter_labels_without_novelty_claim() -> None:
    manifest = _candidate_manifest(load_config(ROOT))
    assert manifest["historical_novelty_claimed"] is False
    assert manifest["post_response_cells"] == 0
    assert manifest["ontology_counts_raw"] == {
        "hidden_matter_required": 131072,
        "baryon_sourced_gravitational_sector": 131072,
    }
    labels = {row["creativity_label"] for row in manifest["niches"]}
    assert labels == {
        "known_formula_control",
        "known_family_control",
        "known_family_combination",
        "potentially_new_synthesis",
    }


def test_item34_real_pool_excludes_all_predecessors_before_balanced_roles() -> None:
    config = load_config(ROOT)
    pool, audit = _fresh_pool(ROOT, config)
    assert len(pool) == 2091
    assert audit["predecessor_union"] == {
        "plateifu_ids": 3558,
        "mangaids": 3542,
        "coordinate_rows": 12023,
    }
    assert audit["exclusions"] == {
        "predecessor_coordinate": 57,
        "predecessor_identity": 3552,
    }
    sample = _sample_manifest(config, pool)
    assert sample["counts"] == {
        "fresh_predictor_pool": 2091,
        "selected": 600,
        "exploration": 480,
        "reserved_confirmation": 120,
        "response_rows_read": 0,
    }
    assert sample["fold_counts_exploration"] == {str(fold): 96 for fold in range(5)}
    assert {
        key: value["eligible"] for key, value in sample["selected_cell_counts"].items()
    } == config["sample"]["expected_cell_capacities"]
    assert all(row["response_read"] is False for row in sample["objects"])


def test_item34_ordinary_coherence_control_is_predictor_only_and_frozen() -> None:
    config = load_config(ROOT)
    pool, _ = _fresh_pool(ROOT, config)
    sample = _sample_manifest(config, pool)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    spec = sample["ordinary_coherence_control_spec"]
    assert spec == _coherence_control_spec(exploration, config)
    design_a = _ordinary_coherence_design(exploration, config, spec)
    design_b = _ordinary_coherence_design(exploration, config, spec)
    assert design_a.shape == (480, 141)
    assert np.array_equal(design_a, design_b)
    assert set(spec) == {
        "phase_space",
        "log_acceleration",
        "log_radius",
        "log_surface_density",
        "age",
    }
    assert all(len(value["knots"]) == 9 for value in spec.values())
    assert "target" not in _ordinary_coherence_design.__annotations__


def test_item34_response_query_contains_two_tracers_and_only_declared_ids() -> None:
    config = load_config(ROOT)
    query = _response_query(config, ["1000-1901", "1001-1902"])
    for column in config["sources"]["response_columns"]:
        assert f"d.{column}" in query
    assert "1000-1901" in query and "1001-1902" in query
    assert "reserved_confirmation" not in query
    assert "halo" not in query.lower()


def test_item34_skyserver_parser_handles_declared_comment_metadata() -> None:
    payload = (
        b"#Table1\nplateifu,stellar_sigma_1re,stellar_rchi2_1re,"
        b"stellar_vel_lo_clip,stellar_vel_hi_clip,ha_gvel_lo_clip,ha_gvel_hi_clip\n"
        b"1000-1901,120,1.2,-90,95,-110,115\n"
    )
    rows, comments = _parse_skyserver_csv(payload)
    assert comments == ["#Table1"]
    assert rows[0]["plateifu"] == "1000-1901"
    assert rows[0]["ha_gvel_hi_clip"] == "115"


def test_item34_Halpha_transfer_reuses_stellar_formula_without_reselection() -> None:
    config = load_config(ROOT)
    folds = np.tile(np.arange(5), 4)
    base = np.linspace(1.8, 2.2, len(folds))
    correction = 0.02 * np.sin(np.arange(len(folds)))
    delta = np.vstack([correction, -correction])
    target = base + correction
    transfer = _transfer_selected_candidates(
        delta, target, base, folds, [0, 0, 0, 0, 0], config, np
    )
    assert transfer["formula_reselection_on_Halpha"] is False
    assert transfer["selected_indices_from_stellar"] == [0, 0, 0, 0, 0]
    assert _mse_for_test(target, transfer["prediction"]) < 1e-28


def _mse_for_test(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((target - prediction) ** 2))


def test_item34_residual_null_preserves_each_response_blind_cell() -> None:
    target = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    reference = np.asarray([0.5, 0.5, 0.5, 5.0, 5.0, 5.0])
    rows = [
        {"sample_cell": "m0-s0"},
        {"sample_cell": "m0-s0"},
        {"sample_cell": "m0-s0"},
        {"sample_cell": "m1-s1"},
        {"sample_cell": "m1-s1"},
        {"sample_cell": "m1-s1"},
    ]
    null = _cell_residual_permutation(
        target, reference, rows, np.random.Generator(np.random.PCG64(340699))
    )
    for indices in (np.asarray([0, 1, 2]), np.asarray([3, 4, 5])):
        assert sorted((null - reference)[indices]) == sorted((target - reference)[indices])


def test_item34_one_empirical_counterexample_is_retained_and_audited_not_vetoed() -> None:
    config = load_config(ROOT)
    target = np.zeros(10)
    reference = np.ones(10)
    candidate = np.zeros(10)
    candidate[-1] = 4.0
    rows = [{"plateifu": f"fixture-{index}"} for index in range(10)]
    audit = _robust_comparison(target, candidate, reference, rows, config)
    assert audit["counterexamples"] == 1
    assert audit["counterexample_fraction"] == 0.1
    assert audit["single_counterexample_is_veto"] is False
    assert audit["full_improvement"] < 0.0
    assert audit["leave_one_most_influential_improvement"] > 0.0
    assert audit["leave_one_changes_improvement_sign"] is True


def test_item34_candidate_decoder_reports_ontology_and_coherence_parameters() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    for niche in range(4):
        index = int(np.where(arrays["niche"] == niche)[0][0])
        record = _candidate_record(index, config, arrays)
        assert record["niche_index"] == niche
        assert record["ontology"] in {
            "hidden_matter_required",
            "baryon_sourced_gravitational_sector",
        }
        assert record["coherence_length_kpc"] > 0.0
        assert record["creativity_label"]
