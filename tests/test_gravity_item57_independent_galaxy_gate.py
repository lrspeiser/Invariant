import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item57_independent_galaxy_gate import (
    CONFIG_PATH,
    GravityItem57Error,
    _parse_predictor_surface_density,
    exponential_disk_velocity_squared,
    gas_disk_velocity_squared,
    parse_photometry_payload,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_unbound_scientific_contract_is_valid_before_freeze() -> None:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    validate_config(ROOT, config, require_bound=False)
    assert config["target_candidate"]["refitting_allowed"] is False
    assert config["little_things"]["new_target_queries_allowed"] == 0
    assert config["confirmation_boundary"]["sparc_confirmation_response_rows_allowed"] == 0


def test_single_counterexample_or_family_pruning_tamper_is_rejected() -> None:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    corrupted = deepcopy(config)
    corrupted["counterexample_policy"]["single_counterexample_terminal"] = True
    with pytest.raises(GravityItem57Error, match="over-pruning"):
        validate_config(ROOT, corrupted, require_bound=False)


def test_photometry_parser_accepts_only_one_expected_row() -> None:
    payload = b"# metadata\nName\tDist\tVMag\tRd\te_Rd\nDDO 126\t4.9\t-14.9\t0.68\t0.05\n"
    row = parse_photometry_payload(payload, expected_name="DDO_126")
    assert row == {
        "name": "DDO 126",
        "distance_mpc": 4.9,
        "absolute_v_magnitude": -14.9,
        "disk_scale_kpc": 0.68,
        "disk_scale_error_kpc": 0.05,
    }


def test_predictor_extractor_returns_only_radius_and_surface_density(tmp_path: Path) -> None:
    path = tmp_path / "predictor.txt"
    path.write_text(
        "# Rarc Rkpc forbidden velocity columns then Sdens\n"
        "10 0.2 999 998 997 996 995 994 993 992 5.0 0.2\n"
        "20 0.4 899 898 897 896 895 894 893 892 4.0 0.3\n",
        encoding="utf-8",
    )
    radius, density = _parse_predictor_surface_density(path)
    np.testing.assert_allclose(radius, [0.2, 0.4])
    np.testing.assert_allclose(density, [5.0, 4.0])


def test_gas_quadrature_is_finite_and_linear_in_mass_normalization() -> None:
    evaluation = np.asarray([0.3, 0.7, 1.1])
    density_radius = np.asarray([0.2, 0.5, 0.8, 1.1, 1.4])
    surface_density = np.asarray([8.0, 7.0, 5.0, 3.0, 1.0])
    low = gas_disk_velocity_squared(
        evaluation,
        density_radius,
        surface_density,
        neutral_gas_factor=1.2,
        softening_kpc=0.08,
        radial_subcells=4,
        azimuthal_cells=96,
        gravitational_constant=4.30091e-6,
    )
    high = gas_disk_velocity_squared(
        evaluation,
        density_radius,
        surface_density,
        neutral_gas_factor=1.52,
        softening_kpc=0.08,
        radial_subcells=4,
        azimuthal_cells=96,
        gravitational_constant=4.30091e-6,
    )
    assert np.all(np.isfinite(low))
    assert np.all(low > 0.0)
    np.testing.assert_allclose(high / low, 1.52 / 1.2, rtol=1.0e-12)


def test_exponential_disk_velocity_squared_scales_with_stellar_mass() -> None:
    radius = np.asarray([0.2, 0.7, 1.4])
    low = exponential_disk_velocity_squared(
        radius,
        disk_mass_msun=1.0e8,
        disk_scale_kpc=0.7,
        gravitational_constant=4.30091e-6,
    )
    high = exponential_disk_velocity_squared(
        radius,
        disk_mass_msun=2.0e8,
        disk_scale_kpc=0.7,
        gravitational_constant=4.30091e-6,
    )
    assert np.all(low > 0.0)
    np.testing.assert_allclose(high, 2.0 * low, rtol=1.0e-12)
