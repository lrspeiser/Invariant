from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item19_massive_carrier import (
    CONFIG_PATH,
    _candidate_digest,
    _point_kernel,
    _yukawa_disk_ratio,
    generate_candidates,
    load_config,
    validate_result,
    verify_sample_freeze,
    verify_science_freeze,
)

ROOT = Path(__file__).resolve().parents[1]


def _raw_config() -> dict:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def test_candidate_generator_is_deterministic_and_preserves_both_niches() -> None:
    config = _raw_config()
    first = generate_candidates(config)
    second = generate_candidates(config)
    assert len(first["family"]) == 183_848
    assert _candidate_digest(first) == _candidate_digest(second)
    assert set(first["family"]) == {0, 1, 2, 3}
    assert np.all(1.0 + first["sign"] * first["alpha_total"] >= 0.05)
    assert np.max(first["maximum_solar_fractional_deviation"]) <= 1.0e-5


def test_normalized_yukawa_is_exactly_the_declared_item16_rewrite() -> None:
    config = _raw_config()
    arrays = generate_candidates(config)
    indices = np.linspace(0, len(arrays["family"]) - 1, 257, dtype=int)
    radii = np.logspace(-9.0, 4.0, 113)[None, :]
    f1 = _point_kernel(radii / arrays["lambda1_kpc"][indices, None])
    f2 = _point_kernel(radii / arrays["lambda2_kpc"][indices, None])
    denominator = 1.0 + arrays["sign"][indices, None] * arrays["alpha_total"][indices, None]
    direct = (
        1.0
        + arrays["sign"][indices, None]
        * (arrays["alpha1"][indices, None] * f1 + arrays["alpha2"][indices, None] * f2)
    ) / denominator
    rewrite = (
        1.0
        - arrays["sign"][indices, None] * arrays["alpha1"][indices, None] / denominator * (1.0 - f1)
        - arrays["sign"][indices, None] * arrays["alpha2"][indices, None] / denominator * (1.0 - f2)
    )
    assert np.max(abs(direct - rewrite)) < 1.0e-14
    assert config["candidate_generator"]["historical_novelty_claimed"] is False
    assert "Item16" in config["candidate_generator"]["known_equivalence"]


def test_attractive_point_carrier_fades_and_disk_kernel_obeys_bounds() -> None:
    config = _raw_config()
    u = np.logspace(-12.0, 12.0, 10_000)
    assert np.all(np.diff(_point_kernel(u)) <= 1.0e-15)
    for x in config["kernel_table"]["component_x_R_over_Rd"]:
        massless, _ = _yukawa_disk_ratio(float(x), 0.0, config)
        intermediate, _ = _yukawa_disk_ratio(float(x), 1.0, config)
        short_range, _ = _yukawa_disk_ratio(float(x), 100.0, config)
        assert abs(massless - 1.0) <= config["kernel_table"]["maximum_massless_relative_error"]
        assert 0.0 < short_range < intermediate < massless + 1.0e-5


def test_repository_freezes_and_result_replay() -> None:
    config = load_config(ROOT)
    verify_science_freeze(ROOT, config)
    verify_sample_freeze(ROOT, config)
    result_path = validate_result(ROOT)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["data_source_receipt"]["confirmation_opened"] == 0
    assert result["frozen_boundary"]["post_response_candidate_cells"] == 0
    assert result["positive_spectral_failure_certificate"]["pass"] is True
    assert result["historical_novelty_claimed"] is False
    assert set(result["publication_track_gates"]) != set(result["gravity_track_gates"])
