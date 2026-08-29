from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item37_alternative_geometry import (
    _action_kernel,
    _add_geometry_source_predictors,
    _admissible_candidates,
    _candidate_deltas,
    _candidate_manifest,
    _candidate_record,
    _contract_digest,
    _fresh_pool,
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


def test_item37_config_preserves_equal_geometry_and_counterexample_boundaries() -> None:
    config = load_config(ROOT)
    assert config["item"] == 37
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
        "known_family_extension": 2,
        "known_formula_family": 1,
        "potentially_new_synthesis": 1,
    }


def test_item37_contract_digest_ignores_only_three_bound_commits() -> None:
    config = load_config(ROOT)
    changed = json.loads(json.dumps(config))
    changed["scientific_freeze_commit"] = "a" * 40
    changed["sample_freeze_commit"] = "b" * 40
    changed["source_feature_freeze_commit"] = "c" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["gates"]["single_object_sensitive_formula_may_promote"] = True
    assert _contract_digest(changed) != _contract_digest(config)


def test_item37_raw_grammar_has_equal_unique_niches() -> None:
    arrays = generate_raw_candidates(load_config(ROOT))
    assert len(arrays["niche"]) == 262144
    assert Counter(arrays["niche"].tolist()) == {0: 65536, 1: 65536, 2: 65536, 3: 65536}
    signatures = np.column_stack([arrays[key] for key in sorted(arrays)])
    assert len(np.unique(signatures, axis=0)) == 262144


def test_item37_admissibility_is_positive_local_and_equivalence_counted() -> None:
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
    assert audit["minimum_admitted_response_eigenvalue"] >= config["admissibility"][
        "minimum_response_eigenvalue"
    ]
    assert audit["maximum_admitted_response_eigenvalue"] <= config["admissibility"][
        "maximum_response_eigenvalue"
    ]
    assert audit["maximum_admitted_local_fractional_geometry_response"] <= config[
        "admissibility"
    ]["maximum_local_fractional_geometry_response"]
    assert audit["behavioral_equivalence_classes_adversarial"] <= audit["admissible_candidates"]


def test_item37_kernels_are_finite_positive_and_recover_local_GR_limit() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    indices = [int(np.where(arrays["niche"] == niche)[0][0]) for niche in range(4)]
    subset = {key: value[indices] for key, value in arrays.items()}
    predictors = {
        "log_acceleration": np.asarray([-13.0, -11.0, -9.0]),
        "source_nonaxisymmetry": np.asarray([0.01, 0.2, 0.7]),
        "mode_frequency_ratio": np.asarray([1.1, 2.0, 2.9]),
        "torsion_proxy": np.asarray([0.1, 0.5, 0.9]),
        "nonmetricity_proxy": np.asarray([0.8, 0.4, 0.1]),
        "Finsler_anisotropy_proxy": np.asarray([0.05, 0.4, 0.8]),
        "affine_holonomy_proxy": np.asarray([0.2, 0.6, 0.3]),
    }
    circular, mode, _, _ = _action_kernel(config, subset, predictors, 0, 4, np)
    delta = _candidate_deltas(config, subset, predictors, 0, 4, np)
    assert circular.shape == (4, 3)
    assert np.all(np.isfinite(delta))
    assert np.all(circular > 0.0)
    assert np.all(mode > 0.0)
    local_circular, local_mode, _, _ = _action_kernel(
        config, subset, _local_predictors(config), 0, 4, np
    )
    assert np.max(np.abs(local_circular - 1.0)) <= 1e-5
    assert np.max(np.abs(local_mode - 1.0)) <= 1e-5


def test_item37_manifest_discloses_GR_equivalence_and_no_novelty_claim() -> None:
    manifest = _candidate_manifest(load_config(ROOT))
    assert manifest["historical_novelty_claimed"] is False
    assert manifest["post_response_cells"] == 0
    assert any("TEGR" in value for value in manifest["equivalence_boundaries"])
    assert len(manifest["equivalence_boundaries"]) >= 6
    assert "target" not in _candidate_deltas.__annotations__


def test_item37_real_pool_excludes_item36_roles_before_mass_quartiles() -> None:
    config = load_config(ROOT)
    pool, audit = _fresh_pool(ROOT, config)
    assert len(pool) == 160
    assert audit["item36_roles"] == 180
    assert audit["additional_coordinate_exclusions"] == 2
    sample = _sample_manifest(config, pool)
    assert sample["counts"] == {
        "fresh_disk_pool": 160,
        "selected": 148,
        "exploration": 120,
        "reserved_confirmation": 28,
        "source_map_rows_read": 0,
        "velocity_response_rows_read": 0,
    }
    assert sample["fold_counts_exploration"] == {str(fold): 24 for fold in range(5)}
    assert {row["sample_cell"] for row in sample["objects"]} == {
        "mq0",
        "mq1",
        "mq2",
        "mq3",
    }
    assert all(row["velocity_response_read"] is False for row in sample["objects"])


def test_item37_geometry_source_proxies_are_response_blind_and_bounded() -> None:
    rows = []
    for index, radius in enumerate((0.5, 1.2)):
        rows.append(
            {
                "annulus_index": index,
                "radius_kpc": radius,
                "weighted_radius_re": radius,
                "enclosed_stellar_mass_msun": 1e9 * radius**1.4,
                "source_nonaxisymmetry": 0.2 + 0.1 * index,
                "axis_ratio": 0.6,
                "centroid_offset_re": 0.1,
            }
        )
    output = _add_geometry_source_predictors(rows)
    assert np.allclose([row["radial_enclosed_mass_slope"] for row in output], 1.4)
    for row in output:
        for label in (
            "torsion_proxy",
            "nonmetricity_proxy",
            "Finsler_anisotropy_proxy",
            "affine_holonomy_proxy",
        ):
            assert 0.0 <= row[label] <= 1.0
        assert "target" not in row


def test_item37_source_control_spec_has_only_frozen_geometry_inputs() -> None:
    config = load_config(ROOT)
    rows = []
    for index in range(20):
        rows.append(
            {
                "log_acceleration": -13.0 + 0.1 * index,
                "weighted_radius_re": 0.2 + 0.05 * index,
                "radial_enclosed_mass_slope": 0.2 + 0.1 * index,
                "mode_frequency_ratio": 1.0 + 0.05 * index,
                "source_nonaxisymmetry": 0.01 + 0.01 * index,
                "torsion_proxy": 0.02 + 0.01 * index,
                "nonmetricity_proxy": 0.03 + 0.01 * index,
                "Finsler_anisotropy_proxy": 0.04 + 0.01 * index,
                "affine_holonomy_proxy": 0.05 + 0.01 * index,
            }
        )
    spec = _source_control_spec(rows, config)
    assert set(spec) == {
        "log_acceleration",
        "weighted_radius_re",
        "radial_enclosed_mass_slope",
        "mode_frequency_ratio",
        "source_nonaxisymmetry",
        "torsion_proxy",
        "nonmetricity_proxy",
        "Finsler_anisotropy_proxy",
        "affine_holonomy_proxy",
    }
    assert all(len(value["knots"]) == 9 for value in spec.values())


def test_item37_screen_recovers_injected_candidate_in_every_fold() -> None:
    config = load_config(ROOT)
    folds = np.repeat(np.arange(5), 4)
    base = np.column_stack((np.linspace(1.5, 2.0, len(folds)), np.full(len(folds), 0.1)))
    delta = np.zeros((3, len(folds), 2))
    delta[0] = 0.02
    delta[1, :, 0] = np.linspace(-0.08, 0.08, len(folds))
    delta[1, :, 1] = np.linspace(0.06, -0.06, len(folds))
    delta[2] = -delta[1]
    target = base + delta[1]
    selected = _screen_joint_candidates(delta, target, base, folds, config, np)
    assert selected["selected_indices"] == [1, 1, 1, 1, 1]
    assert np.mean(np.square(target - selected["prediction"])) < 1e-28


def test_item37_null_permutation_preserves_each_cell_annulus_channel() -> None:
    rows = [{"sample_cell": f"mq{index // 8}", "annulus": f"a{index % 2}"} for index in range(32)]
    target = np.column_stack((np.arange(32, dtype=float), np.arange(32, dtype=float) ** 2))
    reference = np.full((32, 2), 0.5)
    null = _permuted_joint_target(
        target,
        reference,
        rows,
        np.random.Generator(np.random.PCG64(370699)),
    )
    groups = sorted({f"{row['sample_cell']}|{row['annulus']}" for row in rows})
    for channel in range(2):
        for group in groups:
            indices = np.asarray(
                [
                    index
                    for index, row in enumerate(rows)
                    if f"{row['sample_cell']}|{row['annulus']}" == group
                ]
            )
            assert sorted((null - reference)[indices, channel]) == sorted(
                (target - reference)[indices, channel]
            )


def test_item37_one_bad_galaxy_is_sensitive_not_vetoed_or_promotable() -> None:
    config = load_config(ROOT)
    rows = []
    for index in range(10):
        rows.extend({"plateifu": f"fixture-{index}"} for _ in range(2))
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
    assert config["gates"]["single_object_sensitive_formula_may_promote"] is False


def test_item37_candidate_decoder_preserves_action_and_equivalence_labels() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    for niche in range(4):
        index = int(np.where(arrays["niche"] == niche)[0][0])
        record = _candidate_record(index, config, arrays)
        assert record["niche_index"] == niche
        assert record["action_track_eligible"] is True
        assert record["creativity_label"]
        assert record["equivalence_boundary"]
