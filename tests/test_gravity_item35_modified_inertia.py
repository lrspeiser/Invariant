from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item35_modified_inertia import (
    _action_kernel,
    _admissible_candidates,
    _candidate_deltas,
    _candidate_manifest,
    _candidate_record,
    _contract_digest,
    _fresh_pool,
    _harmonic_fit_order_three,
    _local_predictors,
    _permuted_joint_target,
    _robust_joint_by_galaxy,
    _sample_manifest,
    _screen_joint_candidates,
    _source_control_spec,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item35_config_preserves_action_two_track_and_counterexample_boundaries() -> None:
    config = load_config(ROOT)
    assert config["item"] == 35
    assert config["discovery_policy"]["equal_initial_viability"] is True
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["scope"]["paid_api_calls_authorized"] is False
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["gates"]["single_empirical_counterexample_is_veto"] is False
    assert config["gates"]["hard_theoretical_violation_can_be_single_case_veto"] is True
    niches = config["candidate_generator"]["niches"]
    assert sum(bool(row["action_track_eligible"]) for row in niches) == 3
    assert Counter(row["creativity_label"] for row in niches) == {
        "known_formula_control": 1,
        "known_family_extension": 1,
        "known_family_combination": 1,
        "potentially_new_synthesis": 1,
    }


def test_item35_contract_digest_ignores_only_three_bound_commit_ids() -> None:
    config = load_config(ROOT)
    changed = json.loads(json.dumps(config))
    changed["scientific_freeze_commit"] = "a" * 40
    changed["sample_freeze_commit"] = "b" * 40
    changed["source_feature_freeze_commit"] = "c" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["gates"]["single_empirical_counterexample_is_veto"] = True
    assert _contract_digest(changed) != _contract_digest(config)


def test_item35_raw_grammar_has_equal_unique_niches() -> None:
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


def test_item35_admissibility_is_frozen_positive_local_and_behaviorally_counted() -> None:
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
        audit["minimum_admitted_kinetic_eigenvalue"]
        >= config["admissibility"]["minimum_kinetic_eigenvalue"]
    )
    assert (
        audit["maximum_admitted_kinetic_eigenvalue"]
        <= config["admissibility"]["maximum_kinetic_eigenvalue"]
    )
    assert (
        audit["maximum_admitted_local_fractional_inertia_response"]
        <= config["admissibility"]["maximum_local_fractional_inertia_response"]
    )


def test_item35_action_kernel_is_finite_positive_and_locally_shielded() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    indices = [int(np.where(arrays["niche"] == niche)[0][0]) for niche in range(4)]
    subset = {key: value[indices] for key, value in arrays.items()}
    predictors = {
        "log_acceleration": np.asarray([-13.0, -11.0, -9.0]),
        "omega_Gyr_inverse": np.asarray([0.2, 10.0, 200.0]),
        "source_nonaxisymmetry": np.asarray([0.01, 0.2, 0.7]),
        "mode_frequency_ratio": np.asarray([1.0, 2.0, 3.0]),
        "age_gyr_proxy": np.asarray([0.2, 4.0, 9.0]),
        "vertical_to_orbital_frequency": np.asarray([0.2, 1.0, 4.0]),
        "axis_ratio": np.asarray([0.3, 0.6, 0.85]),
    }
    kinetic_circular, kinetic_mode, _, _ = _action_kernel(config, subset, predictors, 0, 4, np)
    delta = _candidate_deltas(config, subset, predictors, 0, 4, np)
    assert kinetic_circular.shape == (4, 3)
    assert np.all(np.isfinite(delta))
    assert np.all(kinetic_circular > 0.0)
    assert np.all(kinetic_mode > 0.0)
    local_circular, local_mode, _, _ = _action_kernel(
        config, subset, _local_predictors(config), 0, 4, np
    )
    assert np.max(np.abs(local_circular - 1.0)) <= 1e-5
    assert np.max(np.abs(local_mode - 1.0)) <= 1e-5


def test_item35_candidate_manifest_labels_equivalence_without_novelty_claim() -> None:
    manifest = _candidate_manifest(load_config(ROOT))
    assert manifest["historical_novelty_claimed"] is False
    assert manifest["post_response_cells"] == 0
    assert len(manifest["equivalence_boundaries"]) >= 4
    assert "target" not in _candidate_deltas.__annotations__


def test_item35_real_pool_excludes_item34_roles_before_balanced_roles() -> None:
    config = load_config(ROOT)
    pool, audit = _fresh_pool(ROOT, config)
    assert len(pool) == 545
    assert audit["item34_roles"] == 600
    assert audit["additional_coordinate_exclusions"] == 11
    assert audit["fresh_before_disk_quality"] == 1480
    sample = _sample_manifest(config, pool)
    assert sample["counts"] == {
        "fresh_disk_pool": 545,
        "selected": 200,
        "exploration": 160,
        "reserved_confirmation": 40,
        "source_map_rows_read": 0,
        "velocity_response_rows_read": 0,
    }
    assert sample["fold_counts_exploration"] == {str(fold): 32 for fold in range(5)}
    assert all(row["velocity_response_read"] is False for row in sample["objects"])


def test_item35_order_three_harmonic_fit_recovers_known_coefficients() -> None:
    theta_degrees = np.linspace(0.0, 359.0, 360)
    theta = np.radians(theta_degrees)
    coefficient = np.asarray([12.0, 100.0, 8.0, -6.0, 4.0, 3.0, -2.0])
    design = np.column_stack(
        [
            np.ones(len(theta)),
            np.cos(theta),
            np.sin(theta),
            np.cos(2.0 * theta),
            np.sin(2.0 * theta),
            np.cos(3.0 * theta),
            np.sin(3.0 * theta),
        ]
    )
    fit = _harmonic_fit_order_three(
        design @ coefficient,
        np.full(len(theta), 0.5),
        theta_degrees,
        np.ones(len(theta)),
        (0.2, 0.8),
        30,
        4,
        20.0,
        1e6,
        2.0,
        0.8,
        "fixture",
        "inner",
    )
    assert np.allclose(fit["coefficients_km_s"], coefficient, atol=1e-10)
    assert abs(fit["circular_speed_km_s"] - 125.0) < 1e-10
    assert fit["noncircular_ratio"] > 0.0


def test_item35_source_control_spec_uses_only_declared_source_predictors() -> None:
    config = load_config(ROOT)
    rows = []
    for index in range(20):
        rows.append(
            {
                "log_acceleration": -13.0 + 0.1 * index,
                "log_omega_Gyr_inverse": -1.0 + 0.1 * index,
                "mode_frequency_ratio": 1.0 + 0.05 * index,
                "source_nonaxisymmetry": 0.01 + 0.01 * index,
                "age_gyr_proxy": 0.5 + 0.2 * index,
                "vertical_to_orbital_frequency": 0.5 + 0.1 * index,
            }
        )
    spec = _source_control_spec(rows, config)
    assert set(spec) == {
        "log_acceleration",
        "log_omega_Gyr_inverse",
        "mode_frequency_ratio",
        "source_nonaxisymmetry",
        "age_gyr_proxy",
        "vertical_to_orbital_frequency",
    }
    assert all(len(value["knots"]) == 9 for value in spec.values())


def test_item35_joint_screen_recovers_an_injected_cell_in_every_fold() -> None:
    config = load_config(ROOT)
    folds = np.repeat(np.arange(5), 4)
    base = np.column_stack([np.linspace(1.5, 2.0, len(folds)), np.linspace(0.02, 0.08, len(folds))])
    delta = np.zeros((3, len(folds), 2))
    delta[0, :, 0] = 0.03
    delta[1, :, 0] = np.linspace(-0.08, 0.08, len(folds))
    delta[1, :, 1] = 0.04 * np.sin(np.arange(len(folds)))
    delta[2] = -delta[1]
    target = base + delta[1]
    selected = _screen_joint_candidates(delta, target, base, folds, config, np)
    assert selected["selected_indices"] == [1, 1, 1, 1, 1]
    assert np.mean(np.square(target - selected["prediction"])) < 1e-28


def test_item35_null_permutation_preserves_each_cell_annulus_channel() -> None:
    rows = [
        {"sample_cell": f"m{index // 8}-n0", "annulus": "inner" if index % 2 == 0 else "outer"}
        for index in range(16)
    ]
    target = np.arange(32, dtype=float).reshape(16, 2)
    reference = np.full((16, 2), 0.5)
    null = _permuted_joint_target(
        target,
        reference,
        rows,
        np.random.Generator(np.random.PCG64(350699)),
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
        for channel in range(2):
            assert sorted((null - reference)[indices, channel]) == sorted(
                (target - reference)[indices, channel]
            )


def test_item35_one_bad_galaxy_is_sensitive_not_a_formula_veto() -> None:
    config = load_config(ROOT)
    rows = []
    for index in range(10):
        rows.extend(
            [
                {"plateifu": f"fixture-{index}", "annulus_index": 0},
                {"plateifu": f"fixture-{index}", "annulus_index": 1},
            ]
        )
    target = np.zeros((20, 2))
    reference = np.ones((20, 2))
    candidate = np.zeros((20, 2))
    candidate[-2:] = 4.0
    audit = _robust_joint_by_galaxy(target, candidate, reference, rows, config)
    assert audit["counterexample_galaxies"] == 1
    assert audit["single_counterexample_is_veto"] is False
    assert audit["full_improvement"] < 0.0
    assert audit["leave_one_most_influential_improvement"] > 0.0
    assert audit["single_object_sensitive"] is True


def test_item35_candidate_decoder_preserves_action_and_creativity_labels() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    for niche in range(4):
        index = int(np.where(arrays["niche"] == niche)[0][0])
        record = _candidate_record(index, config, arrays)
        assert record["niche_index"] == niche
        assert record["creativity_label"]
        assert record["equivalence_boundary"]
        assert record["action_track_eligible"] is (niche != 0)
