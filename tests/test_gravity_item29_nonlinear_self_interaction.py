from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item29_nonlinear_self_interaction import (
    CONFIG_PATH,
    _admissible_candidates,
    _build_log_response_matrix,
    _build_sample,
    _candidate_log_response,
    _candidate_manifest,
    _canonical_identity,
    _coordinates_from_name,
    _short_sky_key,
    _synthetic_controls,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_config(ROOT)


def _synthetic_rows() -> list[dict]:
    rows = []
    for index in range(18):
        mass = 10.0 ** (10.7 + 0.055 * index)
        reff = 2.0 + 0.31 * index
        rows.append(
            {
                "name": f"SL2SJ{index:06d}+010101",
                "fold": index % 3,
                "stellar_mass_proxy_msun": mass,
                "reff_kpc": reff,
                "stellar_surface_density_proxy_msun_kpc2": mass
                / (2.0 * np.pi * reff**2),
                "z_lens": 0.2 + 0.025 * index,
                "z_source": 1.0 + 0.08 * index,
                "g_minus_i": 1.2 + 0.03 * (index % 7),
                "r_minus_z": 0.4 + 0.04 * (index % 5),
                "axis_ratio": 0.55 + 0.025 * (index % 10),
                "g_dyn_m_s2": 10.0 ** (-9.2 - 0.12 * index),
                "g_lens_m_s2": 10.0 ** (-9.0 - 0.10 * index),
                "radius_ratio_dyn": 1.0,
                "radius_ratio_lens": 0.25 + 0.16 * index,
                "sigma_km_s": 180.0 + index,
                "sigma_error_km_s": 10.0,
            }
        )
    return rows


def test_item29_config_is_frozen_to_equal_raw_capacity() -> None:
    config = _config()
    assert config["item"] == 29
    assert config["candidate_generator"]["raw_candidate_cells"] == 262144
    assert [row["raw_cells"] for row in config["candidate_generator"]["niches"]] == [
        65536,
        65536,
        65536,
        65536,
    ]
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["candidate_generator"]["post_response_cells"] == 0


def test_item29_raw_generation_is_deterministic_and_balanced() -> None:
    config = _config()
    first = generate_raw_candidates(config)
    second = generate_raw_candidates(config)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert Counter(first["niche"].tolist()) == Counter({0: 65536, 1: 65536, 2: 65536, 3: 65536})
    for niche in range(4):
        polarity = first["polarity"][first["niche"] == niche]
        assert Counter(polarity.tolist()) == Counter({0: 32768, 1: 32768})


def test_item29_admissibility_is_response_independent_and_preserves_every_niche() -> None:
    config = _config()
    arrays, audit = _admissible_candidates(config)
    generator = config["candidate_generator"]
    assert len(arrays["niche"]) == generator["expected_admissible_candidates"]
    assert audit["admissible_per_niche"] == generator["expected_admissible_per_niche"]
    assert audit["raw_candidate_digest"] == generator["expected_raw_candidate_digest"]
    assert audit["admissible_candidate_digest"] == generator[
        "expected_admissible_candidate_digest"
    ]
    assert audit["filters_are_response_independent"] is True
    assert audit["maximum_admitted_local_fractional_response"] <= 1e-5
    assert all(int(audit["admissible_per_niche"][str(index)]) > 0 for index in range(4))


def test_item29_each_nonlinear_branch_is_finite_on_admissible_candidates() -> None:
    config = _config()
    arrays, _ = _admissible_candidates(config)
    accelerations = np.asarray([[1e-11, 3e-10], [2e-10, 8e-12]])
    radii = np.asarray([[1.0, 0.4], [1.0, 3.0]])
    selected = np.concatenate(
        [np.where(arrays["niche"] == niche)[0][:8] for niche in range(4)]
    )
    subset = {key: value[selected] for key, value in arrays.items()}
    response = _candidate_log_response(
        config, subset, accelerations, radii, 0, len(selected), np
    )
    assert response.shape == (32, 2, 2)
    assert np.all(np.isfinite(response))
    assert np.any(response > 0)
    assert np.any(response < 0)


def test_item29_identity_normalization_catches_truncated_predecessors() -> None:
    assert _canonical_identity("SL2SJ141137+565119") == "141137+565119"
    assert _canonical_identity("SDSSJ222825.76+120503.9") == "22282576+1205039"
    assert _short_sky_key("SDSSJ2228+1205") == "2228+1205"
    assert _short_sky_key("SDSSJ222825.76+120503.9") == "2228+1205"
    ra, dec = _coordinates_from_name("SL2SJ020833-071414")
    assert np.isclose(ra, 32.1375)
    assert np.isclose(dec, -7.2372222222)


def test_item29_sample_roles_are_deterministic_sealed_and_balanced() -> None:
    predictors = []
    for index in range(23):
        predictors.append(
            {
                "name": f"SL2SJ{index:06d}+010101",
                "log10_stellar_mass_msun": 10.5 + index / 20,
            }
        )
    first = _build_sample(predictors, _config())
    second = _build_sample(predictors, _config())
    assert first == second
    assert first["counts"] == {"selected": 23, "exploration": 18, "confirmation": 5}
    assert first["fold_counts"] == {"0": 6, "1": 6, "2": 6}
    assert first["response_values_read"] == 0
    assert first["confirmation_opened"] is False
    assert all(row["fold"] is None for row in first["objects"] if row["role"] == "confirmation")


def test_item29_candidate_manifest_binds_four_injection_niches() -> None:
    manifest = _candidate_manifest(_config())
    assert manifest["synthetic_injection_niches"] == [0, 1, 2, 3]
    assert manifest["post_response_candidate_cells"] == 0
    assert manifest["response_values_read"] == 0


def test_item29_synthetic_controls_recover_the_four_frozen_niches() -> None:
    config = _config()
    arrays, _ = _admissible_candidates(config)
    rows = _synthetic_rows()
    log_response = _build_log_response_matrix(config, arrays, rows, np)
    folds = np.asarray([row["fold"] for row in rows])
    controls = _synthetic_controls(log_response, folds, rows, config, arrays, np)
    assert controls["all_injected_niches_recovered"] is True
    assert controls["GR_control_prefers_nonzero_self_interaction"] is False


def test_item29_config_contract_has_no_response_dependent_candidate_fields() -> None:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    forbidden = {"thetaE", "sigap", "e_sigap"}
    generator_text = json.dumps(config["candidate_generator"], sort_keys=True)
    assert all(value not in generator_text for value in forbidden)
    assert config["sources"]["response_columns"] == [
        "Name",
        "thetaE",
        "sigap",
        "e_sigap",
        "Survey",
    ]
