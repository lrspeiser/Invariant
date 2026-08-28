from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item18_diskmass_antiscreening import (
    _candidate_digest,
    _contract_digest,
    _disk_velocity_sq,
    _prediction_matrix,
    _raw_candidate_count,
    _response_url,
    generate_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item18_candidate_boundary_and_digest_are_frozen() -> None:
    config = load_config(ROOT)
    arrays = generate_candidates(config)
    assert _raw_candidate_count(config) == 750300
    assert len(arrays["amplitude"]) == 621750
    assert (
        _candidate_digest(arrays)
        == "4eff280aac78f18c13cce4bee54ce89a4d4e9c380652a5d01703e7d780c2f235"
    )
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["scope"]["confirmation_opening_authorized"] is False


def test_item18_contract_digest_ignores_only_freeze_bindings() -> None:
    config = load_config(ROOT)
    rebound = json.loads(json.dumps(config))
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    rebound["pre_response_filters"]["maximum_local_fractional_deviation"] = 1e-4
    assert _contract_digest(config) != _contract_digest(rebound)


def test_every_admitted_cell_passes_one_au_and_positive_filters() -> None:
    config = load_config(ROOT)
    arrays = generate_candidates(config)
    constants = config["physics"]["constants"]
    g_sun = constants["G_SI"] * constants["M_sun_kg"] / constants["AU_m"] ** 2
    denominator = 1.0 - arrays["amplitude"] / (1.0 + (g_sun / arrays["a0_m_s2"]) ** arrays["power"])
    nu = 1.0 / denominator
    assert np.max(np.abs(nu - 1.0)) <= 1e-5
    assert np.min(denominator) >= 0.05
    assert np.max(1.0 / (1.0 - arrays["amplitude"])) <= 20.0 + 1e-12


def test_exponential_disk_velocity_has_correct_mass_scaling() -> None:
    first = float(_disk_velocity_sq(1e10, 3.0, 6.6))
    second = float(_disk_velocity_sq(2e10, 3.0, 6.6))
    assert first > 0
    assert math.isclose(second, 2.0 * first, rel_tol=1e-14)


def test_positive_antiscreening_never_reduces_velocity() -> None:
    config = load_config(ROOT)
    arrays = generate_candidates(config)
    sample = {key: value[:2] for key, value in arrays.items()}
    rows = [
        {
            "radius_kpc": 8.0,
            "star_v2_unit": 12000.0,
            "gas_v2_unit": 3000.0,
        }
    ]
    prediction = _prediction_matrix(sample, rows, np)[:, 0]
    gr = 0.5 * np.log(
        sample["stellar_mass_to_light"] * rows[0]["star_v2_unit"]
        + sample["gas_mass_scale"] * rows[0]["gas_v2_unit"]
    )
    assert np.all(prediction >= gr)


def test_response_url_requests_only_one_ugc_and_frozen_columns() -> None:
    config = load_config(ROOT)
    url = _response_url(config, 1234)
    assert "UGC=1234" in url
    assert "Vrot" in url and "hrot" in url
    assert "iiTF" not in url
    assert "HI_width" not in url
