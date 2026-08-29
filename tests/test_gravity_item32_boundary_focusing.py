from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import _read_json
from sigma_theory_compiler.gravity_item32_boundary_focusing import (
    _admissible_candidates,
    _analytic_candidate_response,
    _axial_harmonic_fit,
    _boundary_basis,
    _candidate_manifest,
    _candidate_prediction_matrix,
    _contract_digest,
    _design_matrices,
    _fresh_pool,
    _maps_location,
    _pad_basis,
    _sample_manifest,
    _wrap_axial_degrees,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_map(config: dict) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    axis = np.linspace(-1.8, 1.8, 81)
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    azimuth = np.degrees(np.arctan2(y, x))
    flux = np.exp(-1.7 * radius) * (
        1.0 + 0.22 * np.cos(2.0 * np.radians(azimuth)) + 0.08 * x * y
    )
    flux += 0.12 * np.exp(-((x - 0.45) ** 2 + (y + 0.2) ** 2) / 0.03)
    flux_ivar = np.full_like(flux, 400.0)
    snr = flux * np.sqrt(flux_ivar)
    return _boundary_basis(flux, flux_ivar, snr, radius, azimuth, config)


def test_item32_config_preserves_strict_boundary() -> None:
    config = load_config(ROOT)
    assert config["item"] == 32
    assert config["stable_goal_sha256"] == (
        "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
    )
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["scope"]["paid_api_calls_authorized"] is False
    assert config["sources"]["maps"]["confirmation_download_forbidden"] is True
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["discovery_policy"]["equal_initial_viability"] is True
    assert "stellar_velocity_in_boundary_source_features" in config["scope"]["forbidden_inputs"]


def test_item32_contract_digest_ignores_only_bound_commit_ids() -> None:
    config = load_config(ROOT)
    changed = json.loads(json.dumps(config))
    changed["scientific_freeze_commit"] = "a" * 40
    changed["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["gates"]["maximum_selection_aware_permutation_p"] = 0.5
    assert _contract_digest(changed) != _contract_digest(config)


def test_item32_raw_grammar_has_equal_unique_niches() -> None:
    arrays = generate_raw_candidates(load_config(ROOT))
    assert len(arrays["niche"]) == 262144
    assert Counter(arrays["niche"].tolist()) == {0: 65536, 1: 65536, 2: 65536, 3: 65536}
    signatures = np.column_stack([arrays[key] for key in sorted(arrays)])
    assert len(np.unique(signatures, axis=0)) == 262144
    for niche in range(4):
        selected = arrays["niche"] == niche
        assert np.count_nonzero(arrays["polarity"][selected] == 0) == 32768
        assert np.count_nonzero(arrays["polarity"][selected] == 1) == 32768


def test_item32_admissibility_and_equivalence_counts_are_frozen() -> None:
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
    assert audit["behavioral_duplicate_cells_adversarial"] > 0
    assert set(arrays["niche"].tolist()) == {0, 1, 2, 3}
    assert audit["maximum_admitted_local_fractional_direction_response"] <= config[
        "admissibility"
    ]["maximum_local_fractional_direction_response"]


def test_item32_admitted_equations_are_finite_and_directionally_distinct() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    indices = [int(np.flatnonzero(arrays["niche"] == niche)[0]) for niche in range(4)]
    subset = {key: value[indices] for key, value in arrays.items()}
    response = _analytic_candidate_response(
        config,
        subset,
        np.asarray([0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.4]),
        np.asarray([0.25, 0.45, 0.7, 0.95, 1.15, 1.35, 1.5, 0.6]),
        0,
        4,
    )
    assert response.shape == (4, 8)
    assert np.all(np.isfinite(response))
    assert len(np.unique(np.round(response, 12), axis=0)) == 4


def test_item32_sample_is_fresh_balanced_and_sealed() -> None:
    config = load_config(ROOT)
    pool, prior_ids = _fresh_pool(ROOT, config)
    assert len(prior_ids) == 1200
    assert len(pool) == 52
    assert not ({str(row["plateifu"]) for row in pool} & prior_ids)
    sample = _sample_manifest(config, pool)
    assert sample["counts"] == {
        "fresh_disk_pool": 52,
        "selected": 48,
        "exploration": 40,
        "reserved_confirmation": 8,
        "response_rows_read": 0,
        "map_downloads": 0,
    }
    assert sample["fold_counts_exploration"] == {str(fold): 8 for fold in range(5)}
    assert all(value["eligible"] == 13 for value in sample["selected_cell_counts"].values())
    assert all(row["response_read"] is False for row in sample["objects"])
    assert all(row["map_downloaded"] is False for row in sample["objects"])


def test_item32_maps_location_is_identity_specific() -> None:
    filename, url = _maps_location(load_config(ROOT), "10001-12701")
    assert filename == "manga-10001-12701-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
    assert "/10001/12701/" in url
    assert url.endswith(filename)


def test_item32_axial_fit_recovers_orientation_modulo_spin() -> None:
    azimuth = np.linspace(0.0, 356.0, 90)
    radius = np.tile(np.asarray([0.4, 1.1]), 45)
    velocity = 135.0 * np.cos(np.radians(azimuth - 23.0))
    result = _axial_harmonic_fit(
        velocity,
        radius,
        azimuth,
        np.ones_like(velocity),
        [0.2, 1.5],
        20,
        4,
        20.0,
        1e6,
        "synthetic",
    )
    assert abs(float(result["axial_angle_degrees"]) - 23.0) < 1e-10
    assert abs(float(_wrap_axial_degrees(203.0)) - 23.0) < 1e-12


def test_item32_continuum_basis_has_four_nonduplicate_vector_niches() -> None:
    basis, features = _synthetic_map(load_config(ROOT))
    assert basis["boundary"].shape[0] == 4
    assert basis["direction"].shape[:2] == (4, 4)
    assert len(basis["radius"]) >= 80
    assert set(basis["annulus"].tolist()) == {0, 1}
    assert np.all(np.isfinite(basis["direction"]))
    assert not np.allclose(basis["direction"][0], basis["direction"][2])
    assert features["inner_source_pixels"] >= 40
    assert features["outer_source_pixels"] >= 40
    assert set(load_config(ROOT)["map_source"]["source_features"]) <= set(features)


def test_item32_pixel_evaluator_is_finite_and_uses_no_response() -> None:
    config = load_config(ROOT)
    basis, _ = _synthetic_map(config)
    padded = _pad_basis([("synthetic", basis)])
    arrays, _ = _admissible_candidates(config)
    subset = {key: value[:64] for key, value in arrays.items()}
    prediction = _candidate_prediction_matrix(config, subset, padded, np)
    assert prediction.shape == (64, 1, 2)
    assert np.all(np.isfinite(prediction))
    assert np.max(np.abs(prediction)) <= 90.0


def test_item32_baselines_use_declared_response_blind_columns() -> None:
    config = load_config(ROOT)
    pool, _ = _fresh_pool(ROOT, config)
    sample = _sample_manifest(config, pool)
    rows = []
    for index, row in enumerate(
        value for value in sample["objects"] if value["role"] == "exploration"
    ):
        rows.append(
            {
                **row,
                "source_features": {
                    key: 0.1 * (index + position + 1)
                    for position, key in enumerate(config["map_source"]["source_features"])
                },
            }
        )
    structural, flexible = _design_matrices(rows, config)
    assert structural.shape[:2] == (40, 2)
    assert flexible.shape[:2] == (40, 2)
    assert flexible.shape[-1] > structural.shape[-1]
    assert "plateifu" not in config["evaluation"]["baseline_structural"]
    assert _read_json(ROOT / config["sources"]["item31_sample_manifest"])["counts"][
        "response_rows_read"
    ] == 0


def test_item32_candidate_manifest_never_claims_novelty() -> None:
    manifest = _candidate_manifest(load_config(ROOT))
    assert manifest["historical_novelty_claimed"] is False
    assert manifest["post_response_cells"] == 0
    labels = {row["creativity_label"] for row in manifest["niches"]}
    assert labels == {
        "known_family_extension",
        "known_family_combination",
        "speculative_control",
        "potentially_new_synthesis",
    }
