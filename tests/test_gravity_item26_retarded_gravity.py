from __future__ import annotations

from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item26_retarded_gravity import (
    _admissible_candidates,
    _candidate_values,
    _contract_digest,
    _enclosed_stellar_fraction,
    _log_mu,
    _predictor_rows,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item26_stable_contract_and_equal_raw_capacity() -> None:
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
    assert config["discovery_policy"]["partial_results_are_not_pruned"] is True


def test_item26_candidate_generation_is_deterministic() -> None:
    config = load_config(ROOT)
    first = generate_raw_candidates(config)
    second = generate_raw_candidates(config)
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_item26_admissible_candidates_are_causal_local_and_positive() -> None:
    config = load_config(ROOT)
    arrays, audit, universal = _admissible_candidates(config)
    assert len(arrays["niche"]) == len(universal)
    assert all(int(audit["admissible_niche_counts"][str(niche)]) > 0 for niche in range(4))
    assert audit["advanced_support_cells"] == 0
    assert audit["superluminal_cells"] == 0
    assert audit["maximum_admitted_local_fractional_response"] <= 1e-5
    assert audit["admitted_domain_mu_range"][0] >= 0.05
    assert audit["admitted_domain_mu_range"][1] <= 20.0


def test_item26_static_limit_is_exact() -> None:
    config = load_config(ROOT)
    arrays = generate_raw_candidates(config)
    for begin in range(0, len(arrays["niche"]), 8192):
        end = min(begin + 8192, len(arrays["niche"]))
        values = _candidate_values(config, arrays, begin, end, np)
        result = _log_mu(
            values,
            np.asarray([0.0]),
            np.asarray([1e5]),
            np.asarray([1e-3]),
            np.asarray([1.3]),
            np,
        )
        assert np.max(np.abs(result)) == 0.0


def test_item26_exponential_enclosed_fraction_is_physical() -> None:
    values = [_enclosed_stellar_fraction(radius) for radius in (0.2, 0.8, 1.3, 1.8, 3.0)]
    assert all(0.0 < value < 1.0 for value in values)
    assert values == sorted(values)


def test_item26_contract_digest_ignores_only_bindings() -> None:
    config = load_config(ROOT)
    rebound = dict(config)
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    changed = dict(config)
    changed["hypothesis"] = "changed"
    assert _contract_digest(config) != _contract_digest(changed)


def test_item26_predictor_parser_never_requires_rotation_response() -> None:
    config = load_config(ROOT)
    e1 = [
        {
            "HRS": "1",
            "UGC": "",
            "NGC": "",
            "_RAJ2000": "10.0",
            "_DEJ2000": "20.0",
            "D": "20.0",
            "Type": "Sc",
            "reff": "2.0",
            "D25": "8.0",
            "imag": "12.0",
            "IRAC1": "11.0",
            "logM*": "9.5",
            "E(B-V)": "0.02",
            "VirgoD": "10.0",
            "Memb": "field",
            "HIDef": "0.1",
        }
    ]
    e3 = [
        {
            "HRS": "1",
            "eps": "0.5",
            "imorph": "60",
            "PAmorph": "120",
            "FHa": "2.0",
            "rRC/reff": "2.5",
            "Nbeams": "100",
        }
    ]
    rows, _ = _predictor_rows(
        e1,
        e3,
        {"names": set(), "coordinates": [], "files": 0},
        config,
    )
    assert len(rows) == 1
    assert "velocity_km_s" not in rows[0]
    assert rows[0]["specific_growth_per_year"] > 0.0


def test_item26_result_validates_when_present() -> None:
    config = load_config(ROOT)
    result = ROOT / str(config["paths"]["result"])
    if not result.exists():
        return
    from sigma_theory_compiler.gravity_item26_retarded_gravity import validate_result

    assert validate_result(ROOT) == result
