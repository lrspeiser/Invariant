from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from sigma_theory_compiler.gravity_item27_gravitational_memory import (
    _admissible_candidates,
    _build_term_matrix,
    _candidate_manifest,
    _contract_digest,
    _kernel,
    _kernel_l1,
    _response_summary,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _row() -> dict[str, object]:
    ages = [0.001 + index * 0.35 for index in range(39)]
    fractions = np.linspace(1.0, 2.0, 39)
    fractions /= np.sum(fractions)
    encoded = json.dumps(fractions.tolist())
    return {
        "log_stellar_mass": 10.5,
        "disk_scale_kpc": 3.0,
        "log_SFR": 0.0,
        "mean_log_age_year": 9.5,
        "mean_metallicity_ZH": 0.0,
        "disk_axis_ratio": 0.6,
        "disk_position_angle_deg": 30.0,
        "bulge_fraction_r": 0.2,
        "disk_fraction_r": 0.8,
        "redshift": 0.01,
        "history_ages_gyr": json.dumps(ages),
        "history_global": encoded,
        "history_inner": encoded,
        "history_outer": encoded,
    }


def test_item27_stable_contract_and_equal_raw_capacity() -> None:
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


def test_item27_candidate_generation_is_deterministic() -> None:
    config = load_config(ROOT)
    first = generate_raw_candidates(config)
    second = generate_raw_candidates(config)
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_item27_admitted_kernels_are_positive_normalized_fading_and_integrable() -> None:
    config = load_config(ROOT)
    arrays, audit = _admissible_candidates(config)
    assert len(arrays["niche"]) == audit["admissible_candidates"]
    assert all(int(audit["admissible_niche_counts"][str(niche)]) > 0 for niche in range(4))
    support = np.asarray([0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 14.0])
    for begin in range(0, len(arrays["niche"]), 8192):
        end = min(begin + 8192, len(arrays["niche"]))
        values = {key: value[begin:end] for key, value in arrays.items()}
        kernels = _kernel(values, support, np)
        assert np.max(np.abs(kernels[:, 0] - 1.0)) == 0.0
        assert np.min(kernels) >= 0.0
        assert np.max(np.diff(kernels, axis=1)) <= 1e-12
        assert np.all(np.isfinite(_kernel_l1(values, np)))
    assert audit["advanced_support_cells"] == 0
    assert audit["maximum_admitted_local_fractional_response"] <= 1e-5


def test_item27_frozen_injections_cover_all_niches() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    indices = config["candidate_generator"]["synthetic_injection_admissible_indices"]
    assert [int(arrays["niche"][index]) for index in indices] == [0, 1, 2, 3]
    assert _candidate_manifest(config)["synthetic_injection_admissible_indices"] == indices


def test_item27_zero_history_has_exact_static_limit() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    row = _row()
    zeros = json.dumps([0.0] * 39)
    row["history_global"] = zeros
    row["history_inner"] = zeros
    row["history_outer"] = zeros
    terms = _build_term_matrix(config, arrays, [row], "primary")
    assert np.max(np.abs(terms)) == 0.0


def test_item27_response_parser_uses_only_frozen_v1200_columns() -> None:
    config = load_config(ROOT)
    radius = np.linspace(0.5, 3.1, 75)
    angle = np.linspace(0.0, 2.0 * np.pi, 75, endpoint=False)
    columns = [
        fits.Column(name="BIN_ID", format="J", array=np.arange(75)),
        fits.Column(name="XBIN", format="D", array=10.0 * radius * np.cos(angle)),
        fits.Column(name="YBIN", format="D", array=6.0 * radius * np.sin(angle)),
        fits.Column(name="SNR_BIN", format="D", array=np.full(75, 30.0)),
        fits.Column(name="Vp", format="D", array=120.0 * np.cos(angle)),
        fits.Column(name="DVp", format="D", array=np.full(75, 5.0)),
        fits.Column(name="Sp", format="D", array=np.full(75, 60.0)),
        fits.Column(name="DSp", format="D", array=np.full(75, 8.0)),
        fits.Column(name="QC", format="D", array=np.full(75, 0.05)),
    ]
    hdus = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)])
    stream = io.BytesIO()
    hdus.writeto(stream)
    predictor = _row()
    predictor["disk_scale_arcsec"] = 10.0
    predictor["disk_axis_ratio"] = 0.6
    predictor["disk_position_angle_deg"] = 0.0
    summary, audit = _response_summary(stream.getvalue(), predictor, config)
    assert audit["failure"] is None
    assert summary is not None
    assert summary["primary_vrms_km_s"] > 0.0


def test_item27_contract_digest_ignores_only_commit_bindings() -> None:
    config = load_config(ROOT)
    rebound = dict(config)
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    changed = dict(config)
    changed["hypothesis"] = "changed"
    assert _contract_digest(config) != _contract_digest(changed)


def test_item27_result_validates_when_present() -> None:
    config = load_config(ROOT)
    result = ROOT / str(config["paths"]["result"])
    if not result.exists():
        return
    from sigma_theory_compiler.gravity_item27_gravitational_memory import validate_result

    assert validate_result(ROOT) == result
