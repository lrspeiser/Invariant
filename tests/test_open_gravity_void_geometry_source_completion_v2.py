from __future__ import annotations

import math

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_void_geometry_source_completion_v2 as lane


def test_exact_pickle_trace_becomes_canonical_nonpickle_mask() -> None:
    config = lane.load_config()
    payload, metadata = lane.canonical_mask(config)
    assert len(payload) == 360 * 180
    assert metadata["true_pixels"] == 9133
    assert metadata["pickle_executed"] is False
    assert metadata["payload_sha256"] == config["mask"]["boolean_payload_sha256"]


def test_mask_index_half_open_boundaries_are_exact() -> None:
    assert lane.mask_index(0.0, -90.0) == (0, 0)
    assert lane.mask_index(359.999, 89.999) == (359, 179)
    assert lane.mask_index(360.0, 0.0) == (0, 90)
    with pytest.raises(lane.VoidGeometryV2Error):
        lane.mask_index(0.0, 90.0)


def test_radec_transform_is_right_handed_and_norm_preserving() -> None:
    assert np.allclose(lane.radec_to_xyz(0.0, 0.0, 2.0), [2.0, 0.0, 0.0], atol=1e-15)
    assert np.allclose(lane.radec_to_xyz(90.0, 0.0, 2.0), [0.0, 2.0, 0.0], atol=1e-15)
    assert np.allclose(lane.radec_to_xyz(0.0, 90.0, 2.0), [0.0, 0.0, 2.0], atol=1e-15)
    assert math.isclose(np.linalg.norm(lane.radec_to_xyz(123.0, -31.0, 7.5)), 7.5)


def test_luminosity_distance_is_not_mistaken_for_comoving_distance() -> None:
    config = lane.load_config()
    cosmology = config["distance_contract"]["cosmology"]
    dc = lane._comoving_mpc(0.1, h0=cosmology["H0_km_s_Mpc"], omega_m=cosmology["Omega_m"], c_km_s=cosmology["c_km_s"])
    dl = 1.1 * dc
    z, radius_hinv = lane.luminosity_to_comoving_hinv(dl, config)
    assert z == pytest.approx(0.1, abs=2e-14)
    assert radius_hinv == pytest.approx((cosmology["H0_km_s_Mpc"] / 100.0) * dc)
    assert radius_hinv != pytest.approx((cosmology["H0_km_s_Mpc"] / 100.0) * dl)


def test_receipt_is_deterministic_bounded_and_replays() -> None:
    first, first_payloads = lane.build_receipt()
    second, second_payloads = lane.build_receipt()
    assert first == second and first_payloads == second_payloads
    assert first["content_sha256"] == lane._self_hash(first)
    assert first["scope"]["scientific_rows_decoded"] == 0
    assert "mechanism inference" in first["scope"]["forbidden"]
    assert lane.check_package() == first
