from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item23_bimetric import (
    GravityItem23Error,
    _admissible_candidates,
    _build_sample,
    _candidate_digest,
    _candidate_log_mu,
    _content_hashed,
    _contract_digest,
    _normal_identity,
    _parse_sugohi_csv,
    _sdss_url,
    _solve_monotone_branch,
    _verify_content_hash,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_balances_raw_mechanisms_and_freezes_admissible_digest() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["niche"]) == 262144
    assert np.bincount(raw["niche"]).tolist() == [65536] * 4
    arrays, audit = _admissible_candidates(config)
    assert audit["admissible_cells"] == 168233
    assert audit["admissible_niche_counts"] == {0: 18689, 1: 18472, 2: 65536, 3: 65536}
    assert (
        _candidate_digest(arrays)
        == "a2e5e309ef2c0fb8ed6c4740cce44452919e13e0c245ba06cf7d86622d2ddc11"
    )
    assert audit["maximum_admitted_local_fractional_deviation"] <= 1e-5
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["discovery_policy"]["age_or_history_is_not_privileged"] is True


def test_contract_digest_ignores_only_commit_bindings() -> None:
    config = load_config(ROOT)
    rebound = json.loads(json.dumps(config))
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    rebound["gates"]["phenomenon_minimum_improvement_vs_flexible"] = 0.0
    assert _contract_digest(config) != _contract_digest(rebound)


def test_coordinate_core_normalization_catches_survey_and_precision_rewrites() -> None:
    assert _normal_identity("HSCJ021737.18-051329.4") == _normal_identity("J021737-051329")
    assert _normal_identity("SDSS J015618.12-010747.1") == _normal_identity(
        "HSCJ015618.13-010747.2"
    )


def test_monotone_branch_solver_has_correct_equation_and_limits() -> None:
    x = np.asarray([1e-9, 1e-3, 1.0, 1e3, 1e12])
    c3 = np.full_like(x, 0.5)
    c4 = np.full_like(x, 2.0)
    screen = _solve_monotone_branch(x, c3, c4, 20, np)
    u = screen * x
    assert np.max(np.abs((u + c3 * u**2 + c4 * u**3 - x) / x)) < 1e-9
    assert screen[0] > 0.999
    assert screen[-1] < 1e-7
    assert np.all(np.diff(screen) < 0.0)


def test_candidate_response_links_motion_and_light_with_one_metric_system() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    nonlinear = np.where(arrays["niche"] == 3)[0][0]
    one = {key: value[[nonlinear]] for key, value in arrays.items()}
    rows = [
        {
            "reff_kpc": 3.0,
            "rein_kpc": 5.0,
            "z_luminosity_Lsun": 1.2e11,
        }
    ]
    result = _candidate_log_mu(config, one, rows, 0, 1, np)
    assert result.shape == (1, 1, 2)
    assert np.all(np.isfinite(result))
    assert math.exp(float(result[0, 0, 0])) > 0.0
    assert math.exp(float(result[0, 0, 1])) > 0.0
    assert result[0, 0, 0] >= result[0, 0, 1]


def test_predictor_only_sample_balances_sealed_roles_and_folds() -> None:
    config = load_config(ROOT)
    predictors = [
        {
            "name": f"HSCJ{index:06d}+000000",
            "z_lens": 0.1 + 0.02 * index,
            "z_source": 1.0 + 0.02 * index,
            "z_luminosity_Lsun": 1e11 + index,
            "reff_kpc": 3.0,
            "r_minus_z": 0.5,
            "axis_ratio": 0.8,
            "max_fracflux_rz": 0.01,
            "spectrum_sn": 5.0,
        }
        for index in range(15)
    ]
    sample = _build_sample(predictors, config)
    assert sample["counts"] == {
        "predictor_quality_eligible": 15,
        "selected": 15,
        "exploration": 12,
        "confirmation": 3,
    }
    assert sample["fold_counts"] == {str(index): 3 for index in range(4)}
    for stratum in range(3):
        group = [row for row in sample["objects"] if row["redshift_stratum"] == stratum]
        assert len(group) == 5
        assert sum(row["role"] == "confirmation" for row in group) == 1


def test_spectrum_presence_query_cannot_read_velocity_response() -> None:
    config = load_config(ROOT)
    safe = _sdss_url(config, 29.0, -1.0, False)
    response = _sdss_url(config, 29.0, -1.0, True)
    assert "Vdisp" not in safe
    assert "e_Vdisp" not in safe
    assert "Vdisp" in response
    assert "e_Vdisp" in response


def test_sugohi_csv_parser_normalizes_hash_prefixed_name() -> None:
    body = (
        b"#name,ra,dec,z_lens,z_source,zl_phot,zs_phot,Rein,lens_mag_i,"
        b"source_mag_i,type,discovery,grade,Reference\n"
        b"021737-051329,34.404922,-5.224821,-99,-99,1.23,1.09,1.23,20.21,"
        b"21.90,GG,test,A,test\n"
    )
    rows = _parse_sugohi_csv(body)
    assert rows[0]["name"] == "021737-051329"
    assert rows[0]["Rein"] == "1.23"


def test_content_hash_detects_mutation() -> None:
    payload = _content_hashed({"a": 1, "b": [2, 3]})
    _verify_content_hash(payload, "test")
    payload["a"] = 2
    with pytest.raises(GravityItem23Error, match="changed"):
        _verify_content_hash(payload, "test")
