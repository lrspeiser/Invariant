from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item17_slacs_running_strength import (
    _build_sample,
    _candidate_digest,
    _candidate_log_mu,
    _contract_digest,
    _raw_candidate_count,
    _response_url,
    _screen_log_mu,
    generate_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item17_config_and_filtered_grid_are_frozen() -> None:
    config = load_config(ROOT)
    arrays = generate_candidates(config)
    assert _raw_candidate_count(config) == 127260
    assert len(arrays["amplitude"]) == 118931
    assert (
        _candidate_digest(arrays)
        == "548460c7b703121e762735e5493332e10c4e52b990e27118575db661721b07a6"
    )
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["scope"]["confirmation_opening_authorized"] is False


def test_item17_contract_digest_ignores_only_commit_bindings() -> None:
    config = load_config(ROOT)
    rebound = json.loads(json.dumps(config))
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    rebound["pre_response_filters"]["maximum_local_fractional_deviation"] = 1e-4
    assert _contract_digest(config) != _contract_digest(rebound)


def test_every_admitted_running_cell_passes_local_and_positive_domain_filters() -> None:
    config = load_config(ROOT)
    arrays = generate_candidates(config)

    def response(radius: float) -> np.ndarray:
        logarithm = np.log1p((radius / arrays["r0_kpc"]) ** arrays["power"])
        return 1.0 + arrays["amplitude"] * logarithm / (1.0 + arrays["saturation"] * logarithm)

    local = response(config["pre_response_filters"]["local_radius_kpc"])
    far = response(config["pre_response_filters"]["domain_max_kpc"])
    assert np.max(np.abs(local - 1.0)) <= 1e-5
    assert np.min(local) >= 0.05
    assert np.min(far) >= 0.05


def test_predictor_only_sample_reserves_twelve_and_balances_folds() -> None:
    config = load_config(ROOT)
    bolton = []
    grillo = []
    for index in range(57):
        name = f"J{index:04d}+{index:04d}"
        bolton.append(
            {
                "SDSS": f"{index:08d}+{index:07d}",
                "zFG": f"{0.05 + 0.003 * index:.4f}",
                "zBG": f"{0.5 + 0.004 * index:.4f}",
                "Imag": "17.0",
                "AI": "0.05",
                "L(V555)": "5.0",
                "Re": f"{1.0 + 0.03 * index:.3f}",
                "b/a": "0.75",
                "Mph": "E",
                "Mul": "S",
                "Lens": "A",
                "Name": name,
            }
        )
        base = 5.0 + index
        grillo.append(
            {
                "SLACS": name,
                "MSalBC": str(base * 1.6),
                "MSalM": str(base * 1.5),
                "MChaBC": str(base),
                "MKroM": str(base * 1.1),
            }
        )
    sample = _build_sample(bolton, grillo, config)
    exploration = [row for row in sample if row["role"] == "exploration"]
    confirmation = [row for row in sample if row["role"] == "reserved_confirmation"]
    assert len(exploration) == 45
    assert len(confirmation) == 12
    assert [sum(row["outer_fold"] == fold for row in exploration) for fold in range(5)] == [
        9,
        9,
        9,
        9,
        9,
    ]


def test_running_law_has_identical_matter_and_light_coupling() -> None:
    arrays = {
        "amplitude": np.asarray([1.0]),
        "r0_kpc": np.asarray([1.0]),
        "power": np.asarray([1.0]),
        "saturation": np.asarray([1.0]),
    }
    rows = [{"reff_kpc": 2.0, "rein_kpc": 2.0}]
    log_mu = _candidate_log_mu(arrays, rows, 0, 1, np)
    assert math.isclose(float(log_mu[0, 0, 0]), float(log_mu[0, 0, 1]), rel_tol=0, abs_tol=0)


def test_small_screen_recovers_injected_running_curve() -> None:
    config = load_config(ROOT)
    folds = np.asarray([index % 5 for index in range(15)])
    x = np.linspace(0.0, 1.0, len(folds))
    log_mu = np.stack(
        [
            np.zeros((len(folds), 2)),
            np.column_stack([0.2 * x, 0.15 * x]),
            np.column_stack([0.03 * x**2, 0.3 * x]),
        ]
    )
    y = math.log(1.1) + log_mu[1]
    result = _screen_log_mu(log_mu, y, folds, config, np)
    assert result["selected_indices"] == [1, 1, 1, 1, 1]
    assert np.max(np.abs(result["prediction"] - y)) < 1e-12


def test_response_queries_exclude_mass_targets_and_confirmations_by_construction() -> None:
    config = load_config(ROOT)
    url4 = _response_url(config, "bolton_table4", "SDSS", "00000000+0000000")
    url5 = _response_url(config, "bolton_table5", "Name", "J0000+0000")
    assert "sigma%2Ce_sigma" in url4
    assert "bSIE%2CGood%3F" in url5
    assert "Mtotlen" not in url4 + url5
    assert "fDM" not in url4 + url5
