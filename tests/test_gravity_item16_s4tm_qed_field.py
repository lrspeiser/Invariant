from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    GravityItem16Error,
    _build_sample,
    _candidate_digest,
    _candidate_log_mu,
    _content_hashed,
    _contract_digest,
    _exact_equivalence_classes,
    _parse_vizier_tsv,
    _response_url,
    _screen_log_mu,
    _verify_content_hash,
    generate_candidates,
    hernquist_projected_mass_fraction,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_generator_are_frozen() -> None:
    config = load_config(ROOT)
    arrays = generate_candidates(config)
    assert len(arrays["family1"]) == 262144
    assert (
        _candidate_digest(arrays)
        == "9f7b84f4c47c556ad534b25fe234316d3501ecd22d89d84c0d724da853175765"
    )
    assert _exact_equivalence_classes(arrays) == 261053
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["scope"]["confirmation_opening_authorized"] is False


def test_contract_digest_ignores_only_commit_binding_fields() -> None:
    config = load_config(ROOT)
    rebound = json.loads(json.dumps(config))
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    rebound["gates"]["minimum_joint_mse_improvement_vs_shared_GR"] = 0.0
    assert _contract_digest(config) != _contract_digest(rebound)


def test_vizier_parser_requires_exact_approved_columns() -> None:
    payload = (
        b"# metadata\nTarget\tzL\tzS\n--------------\t------\t------\nSDSSJ0000+0000\t0.1\t0.5\n"
    )
    assert _parse_vizier_tsv(payload, ("Target", "zL", "zS")) == [
        {"Target": "SDSSJ0000+0000", "zL": "0.1", "zS": "0.5"}
    ]


def test_predictor_only_sample_reserves_ten_and_balances_folds() -> None:
    config = load_config(ROOT)
    table1 = []
    table2 = []
    for index in range(40):
        name = f"SDSSJ{index:04d}+{index:04d}"
        table1.append(
            {
                "Target": name,
                "zL": f"{0.05 + 0.005 * index:.4f}",
                "zS": f"{0.5 + 0.005 * index:.4f}",
                "Imag": "17.0",
                "Ai": "0.05",
                "Reff": f"{1 + 0.02 * index:.3f}",
                "q": "0.75",
                "Class": "ESA",
            }
        )
        table2.append({"Target": name, "logM": f"{10.7 + 0.03 * index:.3f}"})
    sample = _build_sample(table1, table2, config)
    exploration = [row for row in sample if row["role"] == "exploration"]
    confirmation = [row for row in sample if row["role"] == "reserved_confirmation"]
    assert len(exploration) == 30
    assert len(confirmation) == 10
    assert sorted(sum(row["outer_fold"] == fold for row in exploration) for fold in range(5)) == [
        6,
        6,
        6,
        6,
        6,
    ]
    assert all(row["outer_fold"] is None for row in confirmation)


def test_hernquist_projected_fraction_has_correct_limits() -> None:
    values = hernquist_projected_mass_fraction(np.asarray([1e-4, 0.1, 1.0, 10.0, 1e4]))
    assert np.all(np.diff(values) > 0)
    assert values[0] < 1e-6
    assert values[2] == 1.0 / 3.0
    assert values[-1] > 0.999


def test_conformal_scalar_changes_matter_but_not_light() -> None:
    config = load_config(ROOT)
    arrays = {
        "family1": np.asarray([2], dtype=np.int16),
        "family2": np.asarray([0], dtype=np.int16),
        "amplitude": np.asarray([6], dtype=np.int16),  # A=1
        "secondary_fraction": np.asarray([0], dtype=np.int16),
        "lambda": np.asarray([10], dtype=np.int16),  # lambda=1 kpc
        "scale_ratio": np.asarray([0], dtype=np.int16),
        "power1": np.asarray([0], dtype=np.int16),
        "power2": np.asarray([0], dtype=np.int16),
        "density_scale": np.asarray([4], dtype=np.int16),
        "density_power": np.asarray([1], dtype=np.int16),
        "polarization1": np.asarray([0], dtype=np.int16),
        "polarization2": np.asarray([0], dtype=np.int16),
    }
    rows = [{"reff_kpc": 1.0, "rein_kpc": 1.0, "surface_density": 1e9}]
    log_mu = _candidate_log_mu(config, arrays, rows, 0, 1, np)
    assert math.isclose(float(np.exp(log_mu[0, 0, 0])), 1.5, rel_tol=1e-12)
    assert math.isclose(float(np.exp(log_mu[0, 0, 1])), 1.0, rel_tol=1e-12)


def test_small_screen_recovers_injected_linked_potential() -> None:
    config = load_config(ROOT)
    folds = np.asarray([index % 5 for index in range(15)])
    x = np.linspace(0.0, 1.0, len(folds))
    log_mu = np.stack(
        [
            np.zeros((len(folds), 2)),
            np.column_stack([0.2 * x, 0.1 * x]),
            np.column_stack([0.05 * x**2, 0.3 * x]),
        ]
    )
    y = math.log(1.2) + log_mu[1]
    result = _screen_log_mu(log_mu, y, folds, config, np)
    assert result["selected_indices"] == [1, 1, 1, 1, 1]
    assert np.max(np.abs(result["prediction"] - y)) < 1e-12


def test_response_queries_name_only_exploration_observables() -> None:
    config = load_config(ROOT)
    url1 = _response_url(config, "table1", "SDSSJ0000+0000")
    url2 = _response_url(config, "table2", "SDSSJ0000+0000")
    assert "Sigma%2Ce_Sigma" in url1
    assert "bSIE" in url2
    assert "logMein" not in url1 + url2
    assert "fDM" not in url1 + url2
    assert "Target=SDSSJ0000%2B0000" in url1


def test_content_hash_detects_mutation() -> None:
    payload = _content_hashed({"a": 1, "b": [2, 3]})
    _verify_content_hash(payload, "test")
    payload["a"] = 2
    with pytest.raises(GravityItem16Error, match="changed"):
        _verify_content_hash(payload, "test")
