from __future__ import annotations

from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item25_time_varying_g import (
    _admissible_candidates,
    _candidate_values,
    _contract_digest,
    _log_mu,
    _predictor_rows,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item25_stable_contract_and_equal_raw_capacity() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["niche"]) == 262144
    assert [int(np.count_nonzero(raw["niche"] == niche)) for niche in range(4)] == [
        65536,
        65536,
        65536,
        65536,
    ]
    assert config["discovery_policy"]["equal_initial_viability"] is True
    assert config["discovery_policy"]["age_or_history_is_not_privileged"] is True
    assert config["discovery_policy"]["paper_claim_requires_unchanged_fresh_replication"] is True


def test_item25_candidate_generation_is_deterministic() -> None:
    config = load_config(ROOT)
    first = generate_raw_candidates(config)
    second = generate_raw_candidates(config)
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_item25_admissible_candidates_pass_external_bounds() -> None:
    config = load_config(ROOT)
    arrays, audit = _admissible_candidates(config)
    assert len(arrays["niche"]) > 0
    assert all(int(audit["admissible_niche_counts"][str(niche)]) > 0 for niche in range(4))
    assert audit["maximum_admitted_absolute_dotG_over_G_per_year"] <= 1e-12
    assert audit["admitted_bbn_mu_range"][0] >= 0.94
    assert audit["admitted_bbn_mu_range"][1] <= 1.05
    assert audit["admitted_recombination_mu_range"][0] >= 0.98
    assert audit["admitted_recombination_mu_range"][1] <= 1.02


def test_item25_all_candidates_normalize_for_mature_local_present() -> None:
    config = load_config(ROOT)
    arrays = generate_raw_candidates(config)
    for begin in range(0, len(arrays["niche"]), 8192):
        end = min(begin + 8192, len(arrays["niche"]))
        values = _candidate_values(config, arrays, begin, end, np)
        result = _log_mu(values, np.asarray([0.0]), np.asarray([1.0]), np)
        assert np.max(np.abs(result)) <= 1e-14


def test_item25_contract_digest_ignores_only_bindings() -> None:
    config = load_config(ROOT)
    rebound = dict(config)
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    changed = dict(config)
    changed["hypothesis"] = "changed"
    assert _contract_digest(config) != _contract_digest(changed)


def test_item25_predictor_parser_never_requires_response() -> None:
    config = load_config(ROOT)
    raw = [
        {"Seq": "1", "z": "1.000", "logM*": "10.50", "logMb": "10.70", "sigma0": "40.0"}
    ]
    rows = _predictor_rows(raw, config)
    assert len(rows) == 1
    assert "vcirc_km_s" not in rows[0]
    assert 0.0 < rows[0]["stellar_fraction"] <= 1.0


def test_item25_result_validates_when_present() -> None:
    config = load_config(ROOT)
    result = ROOT / str(config["paths"]["result"])
    if not result.exists():
        return
    from sigma_theory_compiler.gravity_item25_time_varying_g import validate_result

    assert validate_result(ROOT) == result
