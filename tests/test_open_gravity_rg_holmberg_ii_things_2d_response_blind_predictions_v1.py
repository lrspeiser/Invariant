from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from sigma_theory_compiler import (
    open_gravity_matched_acceleration_cross_scale_predictions_v1 as controls,
)
from sigma_theory_compiler import (
    open_gravity_rg_holmberg_ii_things_2d_response_blind_predictions_v1 as packet,
)


def _header() -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 4
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["NAXIS3"] = 1
    header["NAXIS4"] = 1
    header["BUNIT"] = "METR/SEC"
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL1"] = 124.826344635
    header["CRVAL2"] = 70.7103287499
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 513.0
    header["CDELT1"] = -0.0004166666768
    header["CDELT2"] = 0.0004166666768
    return header


def _metrics(nodes: int) -> dict[str, object]:
    return {
        "nodes_per_axis": nodes,
        "spacing_kpc": 0.25,
        "dimensionless_mass_relative_error": 0.0,
        "source_masses_msun": {"stellar": 1.0, "hi": 1.0, "co": 0.0},
        "total_mass_msun": 2.0,
        "newton_relative_residual": 1e-12,
        "refracted_gravity_solver": {"relative_residual": 1e-12},
        "density_sha256": "1" * 64,
        "epsilon_sha256": "2" * 64,
        "newton_potential_sha256": "3" * 64,
        "refracted_gravity_potential_sha256": "4" * 64,
        "field_sha256": {},
    }


def _source_cell() -> dict[str, object]:
    return {
        "object_id": "UGC04305",
        "conversion_cell_id": "IRAC1_FIXED_ML0P6",
        "geometry": {
            "object_id": "UGC04305",
            "geometry_variant_id": "I38P0",
            "inclination_deg": 38.0,
            "position_angle_deg": 175.0,
            "distance_mpc": 3.4,
            "ra_deg": 124.76541666666664,
            "dec_deg": 70.7235,
        },
        "summary": {"hgas_pc": 200.0, "hstar_pc": 550.0},
        "profile_sha256": "a" * 64,
        "model_lift_label": "MODEL_LIFTED_2P5D",
        "dx_pc": 200.0,
    }


def test_config_and_predecessor_inventory_are_valid() -> None:
    config = packet.load_config(verify_package=False)
    packet.validate_config(config)
    predecessors = packet._load_predecessors(config)
    assert set(predecessors) == {
        "HOLMBERG_II_2D_REPLICATION_PREFLIGHT",
        "SEVEN_HOLDOUT_SOURCE_BUILDER",
        "AUDITED_3D_DST_PCG_MECHANICS",
        "PUBLISHED_CONTROL_FORMULAS",
        "AUDITED_2D_WCS_BEAM_PROJECTION",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("candidate_contract", "a0_m_s2"), 9e-11),
        (("candidate_contract", "response_parameter_fitting"), True),
        (("source_contract", "model_lift_label"), "FULL_3D"),
        (("projection_contract", "response_values_used"), True),
        (("operator_contract", "maximum_local_relative_difference"), 1.0),
        (("claim_boundary", "publication_ready"), True),
    ],
)
def test_material_config_mutations_fail(path: tuple[str, str], value: object) -> None:
    config = copy.deepcopy(packet.load_config(verify_package=False))
    config[path[0]][path[1]] = value
    with pytest.raises(packet.HolmbergPredictionError):
        packet.validate_config(config)


def test_control_arrays_match_published_scalar_formulas() -> None:
    a0 = 1.2e-10
    values = np.asarray([0.0, 1e-13, 1e-11, 1e-9])
    result = packet._candidate_accelerations(values, values * 1.5, a0)
    for index, value in enumerate(values):
        assert result["RAR_2016_ON_NEWTON_3D"][index] == pytest.approx(
            controls.rar_2016(float(value), a0)
        )
        assert result["MOND_STANDARD_MU_ON_NEWTON_3D"][index] == pytest.approx(
            controls.mond_standard(float(value), a0)
        )
    assert np.array_equal(result["NEWTON_3D_DST"], values)
    assert np.array_equal(result["REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"], values * 1.5)


def test_source_intensity_respects_rows_y_columns_x() -> None:
    x_axis = np.asarray([-1000.0, 0.0, 1000.0])
    y_axis = np.asarray([-1000.0, 0.0, 1000.0])
    x, y = np.meshgrid(x_axis, y_axis)
    hi = np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    arrays = {"x_pc": x, "y_pc": y, "hi_surface_msun_pc2": hi}
    sampled = packet._sample_source_intensity(
        arrays,
        np.asarray([[1.0, 0.0]]),
        np.asarray([[0.0, -1.0]]),
    )
    assert sampled.tolist() == [[12.0, 1.0]]


def test_header_contract_is_value_free_and_exact() -> None:
    config = packet.load_config(verify_package=False)
    packet._validate_header(_header(), config["response_header_contract"])
    changed = _header()
    changed["CDELT2"] *= 2.0
    with pytest.raises(packet.HolmbergPredictionError):
        packet._validate_header(changed, config["response_header_contract"])


def test_real_response_loader_uses_headers_not_data(monkeypatch: pytest.MonkeyPatch) -> None:
    config = packet.load_config(verify_package=False)
    predecessors = packet._load_predecessors(config)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("response array decoder was called")

    monkeypatch.setattr(packet.fits, "getdata", forbidden)
    headers = packet._load_response_headers(
        config, predecessors["HOLMBERG_II_2D_REPLICATION_PREFLIGHT"]["_config"]
    )
    assert len(headers) == 4


def test_additional_beam_is_positive_and_normalized() -> None:
    beam = packet.additional_beam(
        (0.00193, 0.0016793, -32.8),
        (0.0038158, 0.0034928, -40.24),
        0.0004166666768,
    )
    assert beam["covariance_minimum_eigenvalue"] > 0.0
    assert beam["kernel_size"] % 2 == 1
    assert np.sum(beam["kernel"]) == pytest.approx(1.0)


def test_synthetic_cell_builds_all_72_map_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    config = packet.load_config(verify_package=False)
    shape = (16, 16)
    marker_newton = np.ones((5, 5))
    marker_rg = np.full((5, 5), 2.0)

    def fake_load(*_args: object, **_kwargs: object) -> dict[str, np.ndarray]:
        axis = np.linspace(-2000.0, 2000.0, 5)
        x, y = np.meshgrid(axis, axis)
        return {
            "stellar_surface_msun_pc2": np.ones((5, 5)),
            "hi_surface_msun_pc2": np.ones((5, 5)),
            "co_surface_msun_pc2": np.zeros((5, 5)),
            "x_pc": x,
            "y_pc": y,
        }

    calls = 0

    def fake_solve(*_args: object, nodes: int, **_kwargs: object):
        nonlocal calls
        calls += 1
        return _metrics(nodes), {
            "NEWTON_3D_DST": (marker_newton, marker_newton),
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG": (marker_rg, marker_rg),
        }

    major = np.tile(np.linspace(-2.0, 2.0, shape[1]), (shape[0], 1))
    disk_y = np.tile(np.linspace(-1.0, 1.0, shape[0])[:, None], (1, shape[1]))
    radius = np.sqrt(major**2 + disk_y**2) + 0.5
    cosine = major / radius

    monkeypatch.setattr(packet, "_load_source_arrays", fake_load)
    monkeypatch.setattr(packet, "_solve_grid", fake_solve)
    monkeypatch.setattr(packet, "_world_grid", lambda _header: (np.zeros(shape), np.zeros(shape)))
    monkeypatch.setattr(
        packet,
        "_disk_sky_coordinates",
        lambda *_args: (major, disk_y, radius, cosine),
    )
    monkeypatch.setattr(packet, "_sample_source_intensity", lambda *_args: np.ones(shape))

    def fake_sample(field, *_args, **_kwargs):
        radial = np.full(shape, float(field[0][0, 0]) * 1e-10)
        return radial, np.zeros(shape)

    monkeypatch.setattr(packet, "_sample_force", fake_sample)
    arrays, diagnostics = packet._build_cell_arrays(config, _source_cell(), {}, {}, _header())
    assert calls == 2
    assert set(arrays) == set(config["private_output"]["array_roles"])
    assert len(arrays) == 13
    assert diagnostics["all_solver_gates_pass"] is True
    assert diagnostics["robust_eligible_pixels"] > 0
    assert diagnostics["natural_eligible_pixels"] > 0
    assert not np.array_equal(
        arrays["NEWTON_3D_DST__ROBUST"],
        arrays["REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG__ROBUST"],
    )


def test_atomic_no_clobber_and_conflict(tmp_path: Path) -> None:
    path = tmp_path / "sealed.bin"
    assert packet._atomic_no_clobber(path, b"same") == "CREATED"
    assert packet._atomic_no_clobber(path, b"same") == "EXISTING_IDENTICAL"
    with pytest.raises(packet.HolmbergPredictionError):
        packet._atomic_no_clobber(path, b"different")


def test_module_contains_no_response_array_decode_call() -> None:
    source = packet._repo_path(packet.MODULE_PATH).read_text(encoding="utf-8")
    assert "fits.getdata" not in source
    assert ".data[" not in source


def test_package_seals_after_finalization() -> None:
    config = packet.load_config()
    assert config["execution_contract"]["candidate_resolution_predictions"] == 72


def test_config_is_valid_json_and_has_exact_13_roles() -> None:
    raw = packet._repo_path(packet.CONFIG_PATH).read_text(encoding="utf-8")
    config = json.loads(raw)
    assert len(config["private_output"]["array_roles"]) == 13
