from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item36_extra_dimensions import (
    _add_radial_source_predictors,
    _admissible_candidates,
    _candidate_delta_log10_speed,
    _candidate_manifest,
    _candidate_record,
    _contract_digest,
    _fresh_pool,
    _gravity_multiplier,
    _local_predictors,
    _permuted_target,
    _robust_by_galaxy,
    _sample_manifest,
    _screen_candidates,
    _source_control_spec,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item36_config_preserves_equal_action_and_counterexample_boundaries() -> None:
    config = load_config(ROOT)
    assert config["item"] == 36
    assert config["discovery_policy"]["equal_initial_viability"] is True
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["scope"]["paid_api_calls_authorized"] is False
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["gates"]["single_empirical_counterexample_is_veto"] is False
    assert config["gates"]["single_object_sensitive_formula_may_promote"] is False
    assert all(
        bool(row["action_track_eligible"]) for row in config["candidate_generator"]["niches"]
    )
    assert Counter(row["creativity_label"] for row in config["candidate_generator"]["niches"]) == {
        "known_family_combination": 2,
        "known_family_extension": 1,
        "potentially_new_synthesis": 1,
    }


def test_item36_contract_digest_ignores_only_three_bound_commits() -> None:
    config = load_config(ROOT)
    changed = json.loads(json.dumps(config))
    changed["scientific_freeze_commit"] = "a" * 40
    changed["sample_freeze_commit"] = "b" * 40
    changed["source_feature_freeze_commit"] = "c" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["gates"]["single_object_sensitive_formula_may_promote"] = True
    assert _contract_digest(changed) != _contract_digest(config)


def test_item36_raw_grammar_has_equal_unique_niches() -> None:
    arrays = generate_raw_candidates(load_config(ROOT))
    assert len(arrays["niche"]) == 262144
    assert Counter(arrays["niche"].tolist()) == {
        0: 65536,
        1: 65536,
        2: 65536,
        3: 65536,
    }
    signatures = np.column_stack([arrays[key] for key in sorted(arrays)])
    assert len(np.unique(signatures, axis=0)) == 262144


def test_item36_admissibility_is_frozen_positive_local_and_equivalence_counted() -> None:
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
    assert (
        audit["minimum_admitted_gravity_multiplier"]
        >= config["admissibility"]["minimum_gravity_multiplier"]
    )
    assert (
        audit["maximum_admitted_gravity_multiplier"]
        <= config["admissibility"]["maximum_gravity_multiplier"]
    )
    assert (
        audit["maximum_admitted_local_fractional_gravity_response"]
        <= config["admissibility"]["maximum_local_fractional_gravity_response"]
    )


def test_item36_kernels_are_finite_positive_and_recover_local_4d_limit() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    indices = [int(np.where(arrays["niche"] == niche)[0][0]) for niche in range(4)]
    subset = {key: value[indices] for key, value in arrays.items()}
    predictors = {
        "radius_kpc": np.asarray([0.1, 1.0, 10.0]),
        "log_acceleration": np.asarray([-13.0, -11.0, -9.0]),
        "source_nonaxisymmetry": np.asarray([0.01, 0.2, 0.7]),
        "radial_source_slope": np.asarray([0.2, 1.0, 2.5]),
        "log_surface_density": np.asarray([6.5, 8.5, 10.5]),
    }
    multiplier, activation = _gravity_multiplier(config, subset, predictors, 0, 4, np)
    delta = _candidate_delta_log10_speed(config, subset, predictors, 0, 4, np)
    assert multiplier.shape == (4, 3)
    assert np.all(np.isfinite(delta))
    assert np.all(multiplier > 0.0)
    assert np.all((activation >= 0.0) & (activation <= 1.0))
    local, _ = _gravity_multiplier(config, subset, _local_predictors(config), 0, 4, np)
    assert np.max(np.abs(local - 1.0)) <= 1e-5


def test_item36_manifest_discloses_screening_and_no_novelty_claim() -> None:
    manifest = _candidate_manifest(load_config(ROOT))
    assert manifest["historical_novelty_claimed"] is False
    assert manifest["post_response_cells"] == 0
    assert "localization" in manifest["four_dimensional_limit"]
    assert len(manifest["equivalence_boundaries"]) >= 5
    assert "target" not in _candidate_delta_log10_speed.__annotations__


def test_item36_real_pool_excludes_item35_roles_before_balanced_roles() -> None:
    config = load_config(ROOT)
    pool, audit = _fresh_pool(ROOT, config)
    assert len(pool) == 342
    assert audit["item35_roles"] == 200
    assert audit["additional_coordinate_exclusions"] == 3
    sample = _sample_manifest(config, pool)
    assert sample["counts"] == {
        "fresh_pool": 342,
        "selected": 180,
        "exploration": 140,
        "reserved_confirmation": 40,
        "source_map_rows_read": 0,
        "velocity_response_rows_read": 0,
    }
    assert sample["fold_counts_exploration"] == {str(fold): 28 for fold in range(5)}
    assert all(row["velocity_response_read"] is False for row in sample["objects"])


def test_item36_radial_source_slopes_are_derived_without_responses() -> None:
    rows = []
    for index, radius in enumerate((0.3, 0.6, 0.95, 1.3)):
        rows.append(
            {
                "annulus_index": index,
                "radius_kpc": radius,
                "enclosed_stellar_mass_msun": 1e9 * radius**1.5,
                "source_nonaxisymmetry": 0.1,
                "log_surface_density": 8.0,
            }
        )
    output = _add_radial_source_predictors(rows)
    assert np.allclose([row["radial_source_slope"] for row in output], 1.5)
    assert all(0.0 <= row["localization_coordinate"] <= 1.0 for row in output)
    assert "target" not in output[0]


def test_item36_source_control_spec_has_only_frozen_radial_inputs() -> None:
    config = load_config(ROOT)
    rows = []
    for index in range(20):
        rows.append(
            {
                "weighted_radius_re": 0.2 + 0.05 * index,
                "radius_kpc": 0.3 + 0.2 * index,
                "log_acceleration": -13.0 + 0.1 * index,
                "radial_source_slope": 0.2 + 0.1 * index,
                "source_nonaxisymmetry": 0.01 + 0.01 * index,
                "localization_coordinate": 0.02 + 0.02 * index,
            }
        )
    spec = _source_control_spec(rows, config)
    assert set(spec) == {
        "weighted_radius_re",
        "radius_kpc",
        "log_acceleration",
        "radial_source_slope",
        "source_nonaxisymmetry",
        "localization_coordinate",
    }
    assert all(len(value["knots"]) == 9 for value in spec.values())


def test_item36_screen_recovers_injected_candidate_in_every_fold() -> None:
    config = load_config(ROOT)
    folds = np.repeat(np.arange(5), 4)
    base = np.linspace(1.5, 2.0, len(folds))
    delta = np.zeros((3, len(folds)))
    delta[0] = 0.02
    delta[1] = np.linspace(-0.08, 0.08, len(folds))
    delta[2] = -delta[1]
    target = base + delta[1]
    selected = _screen_candidates(delta, target, base, folds, config, np)
    assert selected["selected_indices"] == [1, 1, 1, 1, 1]
    assert np.mean(np.square(target - selected["prediction"])) < 1e-28


def test_item36_null_permutation_preserves_each_cell_annulus() -> None:
    rows = [{"sample_cell": f"m{index // 8}-r0", "annulus": f"a{index % 4}"} for index in range(32)]
    target = np.arange(32, dtype=float)
    reference = np.full(32, 0.5)
    null = _permuted_target(
        target,
        reference,
        rows,
        np.random.Generator(np.random.PCG64(360699)),
    )
    groups = sorted({f"{row['sample_cell']}|{row['annulus']}" for row in rows})
    for group in groups:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if f"{row['sample_cell']}|{row['annulus']}" == group
            ]
        )
        assert sorted((null - reference)[indices]) == sorted((target - reference)[indices])


def test_item36_one_bad_galaxy_is_sensitive_not_vetoed_or_promotable() -> None:
    config = load_config(ROOT)
    rows = []
    for index in range(10):
        rows.extend({"plateifu": f"fixture-{index}"} for _ in range(4))
    target = np.zeros(40)
    reference = np.ones(40)
    candidate = np.zeros(40)
    candidate[-4:] = 4.0
    audit = _robust_by_galaxy(target, candidate, reference, rows, config)
    assert audit["counterexample_galaxies"] == 1
    assert audit["single_counterexample_is_veto"] is False
    assert audit["full_improvement"] < 0.0
    assert audit["leave_one_most_influential_improvement"] > 0.0
    assert audit["single_object_sensitive"] is True


def test_item36_candidate_decoder_preserves_action_and_equivalence_labels() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    for niche in range(4):
        index = int(np.where(arrays["niche"] == niche)[0][0])
        record = _candidate_record(index, config, arrays)
        assert record["niche_index"] == niche
        assert record["action_track_eligible"] is True
        assert record["creativity_label"]
        assert record["equivalence_boundary"]
