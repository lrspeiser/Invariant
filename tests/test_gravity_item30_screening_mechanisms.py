from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item30_screening_mechanisms import (
    _admissible_candidates,
    _candidate_activation,
    _candidate_delta_log10_sigma,
    _candidate_manifest,
    _candidate_values,
    _contract_digest,
    _response_query,
    _sample_manifest,
    generate_raw_candidates,
    load_config,
    prepare_predictors,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item30_config_preserves_strict_boundary() -> None:
    config = load_config(ROOT)
    assert config["item"] == 30
    assert config["stable_goal_sha256"] == (
        "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
    )
    assert config["discovery_policy"]["equal_initial_viability"] is True
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["scope"]["paid_api_calls_authorized"] is False
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert "halo_mass" in config["scope"]["forbidden_inputs"]
    assert config["sources"]["manga_predictor_response_columns_read"] == 0


def test_item30_contract_digest_ignores_only_bound_commit_ids() -> None:
    config = load_config(ROOT)
    changed = json.loads(json.dumps(config))
    changed["scientific_freeze_commit"] = "a" * 40
    changed["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["gates"]["maximum_selection_aware_permutation_p"] = 0.5
    assert _contract_digest(changed) != _contract_digest(config)


def test_item30_raw_grammar_has_exact_equal_unique_niches() -> None:
    config = load_config(ROOT)
    arrays = generate_raw_candidates(config)
    assert len(arrays["niche"]) == 262144
    assert Counter(arrays["niche"].tolist()) == {0: 65536, 1: 65536, 2: 65536, 3: 65536}
    signatures = np.column_stack([arrays[key] for key in sorted(arrays)])
    assert len(np.unique(signatures, axis=0)) == 262144
    for niche in range(4):
        mask = arrays["niche"] == niche
        polarity = arrays["polarity"][mask]
        assert np.count_nonzero(polarity == 0) == 32768
        assert np.count_nonzero(polarity == 1) == 32768


def test_item30_admissibility_is_frozen_local_and_positive() -> None:
    config = load_config(ROOT)
    arrays, audit = _admissible_candidates(config)
    generator = config["candidate_generator"]
    assert audit["raw_candidate_digest"] == generator["expected_raw_candidate_digest"]
    assert audit["admissible_candidate_digest"] == generator["expected_admissible_candidate_digest"]
    assert audit["admissible_candidates"] == generator["expected_admissible_candidates"]
    assert audit["admissible_per_niche"] == generator["expected_admissible_per_niche"]
    assert set(arrays["niche"].tolist()) == {0, 1, 2, 3}
    assert audit["minimum_admitted_mu"] > 0.0
    assert (
        audit["maximum_admitted_local_fractional_response"]
        <= config["admissibility"]["maximum_local_fractional_response"]
    )


def test_item30_all_family_equations_are_finite_and_target_blind() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    indices = [int(np.where(arrays["niche"] == niche)[0][0]) for niche in range(4)]
    subset = {key: value[indices] for key, value in arrays.items()}
    predictors = {
        "dimensionless_potential": np.asarray([1e-9, 1e-7, 1e-5]),
        "density_msun_kpc3": np.asarray([1e5, 1e8, 1e11]),
        "mass_msun": np.asarray([1e8, 1e10, 1e12]),
        "radius_kpc": np.asarray([20.0, 5.0, 0.2]),
        "q_lss": np.asarray([-6.0, -2.0, 1.0]),
        "eta_k": np.asarray([-1.0, 0.0, 2.0]),
    }
    activation = _candidate_activation(config, subset, predictors, 0, 4, np)
    delta = _candidate_delta_log10_sigma(config, subset, predictors, 0, 4, np)
    assert activation.shape == (4, 3)
    assert np.all(np.isfinite(activation))
    assert np.all((activation >= 0.0) & (activation <= 1.0))
    assert np.all(np.isfinite(delta))
    # Neither evaluator accepts a velocity or target argument.
    assert "target" not in _candidate_activation.__annotations__


def test_item30_candidate_labels_do_not_prune() -> None:
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


def test_item30_sample_builder_balances_roles_and_folds() -> None:
    config = load_config(ROOT)
    rows = []
    # Deliberately provide more than the frozen 250 objects in every predictor cell.
    for phi_bin in range(2):
        for environment_bin in range(2):
            for ordinal in range(270):
                rows.append(
                    {
                        "plateifu": f"{phi_bin}{environment_bin}{ordinal:04d}-1901",
                        "mangaid": f"{phi_bin}-{environment_bin}-{ordinal}",
                        "log_dimensionless_potential": -8.0 + 3.0 * phi_bin + ordinal * 1e-8,
                        "gema_q_lss": -5.0 + 6.0 * environment_bin + ordinal * 1e-8,
                    }
                )
    sample = _sample_manifest(config, rows)
    roles = Counter(row["role"] for row in sample["objects"])
    assert roles == {"exploration": 800, "reserved_confirmation": 200}
    assert sample["fold_counts_exploration"] == {str(fold): 160 for fold in range(5)}
    assert all(row["response_read"] is False for row in sample["objects"])


def test_item30_response_query_contains_only_declared_exploration_ids() -> None:
    config = load_config(ROOT)
    query = _response_query(config, ["1000-1901", "1001-1902"])
    for column in config["sources"]["response_columns"]:
        assert f"d.{column}" in query
    assert "1000-1901" in query and "1001-1902" in query
    assert "stellar_sigma_1re" in query
    assert "reserved_confirmation" not in query
    assert "dark" not in query.lower()


def test_item30_frozen_predecessor_audit_is_response_blind() -> None:
    path = (
        ROOT / "runs/gravity/roadmap/item-30-screening-mechanisms-v1-source/predecessor-audit.json"
    )
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["response_values_read"] == 0
    assert audit["paid_api_calls"] == 0
    assert audit["total_coordinate_rows"] > 10000
    assert audit["unique_manga_identities"] == 4260
    assert audit["item24_role_coordinates"] == 160


def test_item30_inherited_predictor_receipt_uses_legacy_digest_convention(
    monkeypatch,
) -> None:
    def stop_before_source_access(*_args, **_kwargs):
        raise RuntimeError("stop after inherited predictor verification")

    monkeypatch.setattr(
        "sigma_theory_compiler.gravity_item30_screening_mechanisms.verify_science_freeze",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "sigma_theory_compiler.gravity_item30_screening_mechanisms._minimum_separations_arcsec",
        stop_before_source_access,
    )
    try:
        prepare_predictors(ROOT)
    except RuntimeError as error:
        assert str(error) == "stop after inherited predictor verification"
    else:
        raise AssertionError("test sentinel did not stop predictor preparation")


def test_item30_candidate_value_decoder_reports_physical_parameters() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    values = _candidate_values(config, arrays, 0, 3, np)
    assert values["polarity"].shape == (3,)
    assert np.all(np.isin(values["polarity"], [-1.0, 1.0]))
    assert np.all(values["amplitude"] > 0.0)
