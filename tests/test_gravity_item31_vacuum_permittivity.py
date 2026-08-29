from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import _read_json, _read_tsv
from sigma_theory_compiler.gravity_item31_vacuum_permittivity import (
    _admissible_candidates,
    _candidate_activation,
    _candidate_delta_log10_sigma,
    _candidate_manifest,
    _candidate_values,
    _contract_digest,
    _parse_skyserver_csv,
    _response_query,
    _sample_manifest,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item31_config_preserves_strict_boundary() -> None:
    config = load_config(ROOT)
    assert config["item"] == 31
    assert config["stable_goal_sha256"] == (
        "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
    )
    assert config["discovery_policy"]["equal_initial_viability"] is True
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["scope"]["paid_api_calls_authorized"] is False
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert "halo_mass" in config["scope"]["forbidden_inputs"]
    assert config["sources"]["inherited_predictor_response_columns_read"] == 0


def test_item31_contract_digest_ignores_only_bound_commit_ids() -> None:
    config = load_config(ROOT)
    changed = json.loads(json.dumps(config))
    changed["scientific_freeze_commit"] = "a" * 40
    changed["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["gates"]["maximum_selection_aware_permutation_p"] = 0.5
    assert _contract_digest(changed) != _contract_digest(config)


def test_item31_raw_grammar_has_exact_equal_unique_niches() -> None:
    arrays = generate_raw_candidates(load_config(ROOT))
    assert len(arrays["niche"]) == 262144
    assert Counter(arrays["niche"].tolist()) == {0: 65536, 1: 65536, 2: 65536, 3: 65536}
    signatures = np.column_stack([arrays[key] for key in sorted(arrays)])
    assert len(np.unique(signatures, axis=0)) == 262144
    for niche in range(4):
        mask = arrays["niche"] == niche
        assert np.count_nonzero(arrays["polarity"][mask] == 0) == 32768
        assert np.count_nonzero(arrays["polarity"][mask] == 1) == 32768


def test_item31_admissibility_is_frozen_local_positive_and_directional() -> None:
    config = load_config(ROOT)
    arrays, audit = _admissible_candidates(config)
    generator = config["candidate_generator"]
    assert audit["raw_candidate_digest"] == generator["expected_raw_candidate_digest"]
    assert audit["admissible_candidate_digest"] == generator["expected_admissible_candidate_digest"]
    assert audit["admissible_candidates"] == generator["expected_admissible_candidates"]
    assert audit["admissible_per_niche"] == generator["expected_admissible_per_niche"]
    assert set(arrays["niche"].tolist()) == {0, 1, 2, 3}
    assert audit["minimum_admitted_mu"] > 0.0
    assert audit["minimum_admitted_direction_contrast"] > 0.0
    assert 0 < audit["behavioral_equivalence_classes_adversarial"] <= len(arrays["niche"])
    assert audit["behavioral_duplicate_cells_adversarial"] >= 0
    assert (
        audit["maximum_admitted_local_fractional_response"]
        <= config["admissibility"]["maximum_local_fractional_response"]
    )


def test_item31_all_equations_are_finite_bounded_and_target_blind() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    indices = [int(np.where(arrays["niche"] == niche)[0][0]) for niche in range(4)]
    subset = {key: value[indices] for key, value in arrays.items()}
    predictors = {
        "acceleration_m_s2": np.asarray([1e-13, 2e-10, 1e-5]),
        "density_msun_kpc3": np.asarray([1e4, 1e8, 1e12]),
        "q_lss": np.asarray([-6.0, -2.0, 1.0]),
        "eta_k": np.asarray([-1.0, 0.0, 2.0]),
        "dnn_mpc": np.asarray([3.0, 0.5, 0.01]),
    }
    activation = _candidate_activation(config, subset, predictors, 0, 4, np)
    delta = _candidate_delta_log10_sigma(config, subset, predictors, 0, 4, np)
    assert activation.shape == (4, 3)
    assert np.all(np.isfinite(activation))
    assert np.all((activation >= 0.0) & (activation <= 1.0))
    assert np.all(np.isfinite(delta))
    assert "target" not in _candidate_activation.__annotations__


def test_item31_candidate_labels_preserve_creative_branch_without_claiming_novelty() -> None:
    manifest = _candidate_manifest(load_config(ROOT))
    assert manifest["historical_novelty_claimed"] is False
    assert manifest["post_response_cells"] == 0
    labels = {row["creativity_label"] for row in manifest["niches"]}
    assert labels == {
        "known_family_control",
        "known_family_extension",
        "potentially_new_synthesis",
    }
    assert manifest["audit"]["raw_per_niche"] == {
        "0": 65536,
        "1": 65536,
        "2": 65536,
        "3": 65536,
    }


def test_item31_real_response_blind_pool_excludes_all_item30_roles() -> None:
    config = load_config(ROOT)
    inherited = _read_tsv(ROOT / config["sources"]["inherited_predictors"])
    predecessor = _read_json(ROOT / config["sources"]["item30_sample_manifest"])
    prior_ids = {str(row["plateifu"]) for row in predecessor["objects"]}
    pool = [
        row
        for row in inherited
        if str(row["plateifu"]) not in prior_ids
        and float(row["snr_med_g"]) >= float(config["sample"]["predictor_minimum_snr_med_g"])
    ]
    assert len(prior_ids) == 1000
    assert len(pool) == 345
    assert not ({str(row["plateifu"]) for row in pool} & prior_ids)
    sample = _sample_manifest(config, pool)
    assert sample["counts"] == {
        "fresh_predictor_pool": 345,
        "selected": 200,
        "exploration": 160,
        "reserved_confirmation": 40,
        "response_rows_read": 0,
    }
    assert sample["fold_counts_exploration"] == {str(fold): 32 for fold in range(5)}
    assert all(row["response_read"] is False for row in sample["objects"])


def test_item31_response_query_contains_only_declared_ids_and_columns() -> None:
    config = load_config(ROOT)
    query = _response_query(config, ["1000-1901", "1001-1902"])
    for column in config["sources"]["response_columns"]:
        assert f"d.{column}" in query
    assert "1000-1901" in query and "1001-1902" in query
    assert "reserved_confirmation" not in query
    assert "dark" not in query.lower()


def test_item31_skyserver_parser_handles_declared_comment_metadata() -> None:
    payload = (
        b"#Table1\nplateifu,stellar_sigma_1re,stellar_rchi2_1re,"
        b"stellar_vel_lo_clip,stellar_vel_hi_clip\n"
        b"1000-1901,120,1.2,-90,95\n"
    )
    rows, comments = _parse_skyserver_csv(payload)
    assert comments == ["#Table1"]
    assert rows == [
        {
            "plateifu": "1000-1901",
            "stellar_sigma_1re": "120",
            "stellar_rchi2_1re": "1.2",
            "stellar_vel_lo_clip": "-90",
            "stellar_vel_hi_clip": "95",
        }
    ]


def test_item31_candidate_decoder_reports_physical_parameters() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    values = _candidate_values(config, arrays, 0, 3, np)
    assert values["polarity"].shape == (3,)
    assert np.all(np.isin(values["polarity"], [-1.0, 1.0]))
    assert np.all(values["amplitude"] > 0.0)
    assert np.all(np.isfinite(values["acceleration_threshold"]))
    assert np.all(np.isfinite(values["density_threshold"]))
