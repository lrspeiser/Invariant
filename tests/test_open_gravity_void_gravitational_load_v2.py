from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_void_gravitational_load_v2 as lane


def raw_config() -> dict:
    return json.loads(lane.CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_file_and_semantic_pins_are_exact() -> None:
    config = raw_config()
    assert lane.file_sha256(lane.CONFIG_PATH) == lane._CONFIG_RAW_SHA256
    assert lane.content_sha256(config) == lane._CONFIG_CONTENT_SHA256
    assert lane.module_semantic_sha256() == lane._MODULE_SEMANTIC_SHA256
    assert lane.file_sha256(lane.TEST_PATH) == lane._TEST_RAW_SHA256
    lane.validate_config(config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("claim_boundary", "real_data_fit"), True),
        (("access_accounting", "scientific_rows_decoded_by_this_packet"), 1),
        (("parameters", "photon_eta"), float("nan")),
        (("real_data_contract", "status"), "READY"),
    ],
)
def test_semantic_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(raw_config())
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises((lane.VoidLoadV2Error, ValueError)):
        lane.validate_config(config)


def test_fixture_registry_is_genuinely_three_dimensional() -> None:
    fixtures = lane.fixture_densities(raw_config())
    assert set(fixtures) == {
        "HOMOGENEOUS",
        "VOID_WITH_TWO_OFF_AXIS_SOURCES",
        "ASYMMETRIC_BAR_AND_CLUMP",
        "ZERO_SOURCE",
    }
    assert all(value.shape == (9, 9, 9) for value in fixtures.values())
    assert np.max(fixtures["VOID_WITH_TWO_OFF_AXIS_SOURCES"][:, 4, 4]) < np.max(
        fixtures["VOID_WITH_TWO_OFF_AXIS_SOURCES"]
    )


def test_vq02_clock_to_space_conversion_occurs_once() -> None:
    q = np.array([0.2, 0.5, 0.9, 0.4], dtype=float)
    direct = lane.direct_log_shift(q, eta=0.3, c=2.0, dx=0.5)
    slowed, delay = lane.slowed_ray_observables(q, eta=0.3, beta=0.0, c=2.0, dx=0.5)
    assert slowed == direct
    assert delay == 0.0
    nonlinear, delay = lane.slowed_ray_observables(q, eta=0.3, beta=0.4, c=2.0, dx=0.5)
    expected = -0.3 / 2.0 * lane.path_integral(q * (1.0 + 0.4 * q), 0.5)
    assert nonlinear == pytest.approx(expected, abs=1e-15)
    assert delay > 0.0


def test_vq04_solves_the_declared_3d_helmholtz_operator_and_l0() -> None:
    config = raw_config()
    rho = lane.fixture_densities(config)["ASYMMETRIC_BAR_AND_CLUMP"]
    p = config["parameters"]
    field = lane.helmholtz_feed(rho, amplitude=p["A_feed"], length=p["L_g"], dx=1.0)
    assert (
        lane.helmholtz_spectral_residual(field, rho, amplitude=p["A_feed"], length=p["L_g"], dx=1.0)
        < 1e-12
    )
    assert np.array_equal(
        lane.helmholtz_feed(rho, amplitude=p["A_feed"], length=0.0, dx=1.0),
        p["A_feed"] * rho,
    )
    assert not np.allclose(field[:, 4, 4], field[:, 3, 4])


def test_vq05_is_a_3d_ivp_with_exact_d0_boundary() -> None:
    config = raw_config()
    rho = lane.fixture_densities(config)["VOID_WITH_TWO_OFF_AXIS_SOURCES"]
    p = config["parameters"]
    source = p["A_feed"] * rho
    expected = lane.local_equilibrium(rho, source=source, gamma0=p["Gamma_0"], gamma_b=p["Gamma_b"])
    observed = lane.solve_diffusive_reservoir(
        rho,
        source=source,
        diffusivity=0.0,
        gamma0=p["Gamma_0"],
        gamma_b=p["Gamma_b"],
        dx=1.0,
        dt=p["diffusion_dt"],
        steps=p["diffusion_steps"],
    )
    assert np.array_equal(observed, expected)
    evolved = lane.solve_diffusive_reservoir(
        rho,
        source=source,
        diffusivity=p["D_g"],
        gamma0=p["Gamma_0"],
        gamma_b=p["Gamma_b"],
        dx=1.0,
        dt=p["diffusion_dt"],
        steps=p["diffusion_steps"],
    )
    assert not np.array_equal(evolved, expected)
    residual = p["D_g"] * lane.periodic_laplacian(evolved, 1.0)
    residual += source - (p["Gamma_0"] + p["Gamma_b"] * rho) * evolved
    assert np.max(np.abs(residual)) < 2e-5


def test_vq06_uses_off_axis_three_dimensional_columns_and_sigma_boundary() -> None:
    rho = np.zeros((9, 9, 9), dtype=float)
    rho[1, 1, 1] = 2.0
    rho[7, 7, 7] = 1.0
    unattenuated = lane.column_attenuated_feed(
        rho, amplitude=1.1, length=1.6, sigma=0.0, dx=1.0, samples=11
    )
    attenuated = lane.column_attenuated_feed(
        rho, amplitude=1.1, length=1.6, sigma=0.25, dx=1.0, samples=11
    )
    assert np.all(np.isfinite(unattenuated))
    assert np.max(unattenuated) > 0.0
    assert np.any(attenuated < unattenuated)
    rotated = np.rot90(rho, axes=(1, 2))
    rotated_result = lane.column_attenuated_feed(
        rotated, amplitude=1.1, length=1.6, sigma=0.25, dx=1.0, samples=11
    )
    assert np.allclose(rotated_result, np.rot90(attenuated, axes=(1, 2)), atol=1e-13)


def test_static_exposure_reverses_but_memory_retains_order() -> None:
    q = np.array([0.1, 0.2, 1.4, 0.3, 0.8], dtype=float)
    rho = np.array([0.8, 0.1, 1.2, 0.4, 0.2], dtype=float)
    assert lane.path_integral(q, 1.0) == lane.path_integral(q[::-1], 1.0)
    forward, _ = lane.photon_memory_log_shift(
        q, rho, drive=0.5, relax0=0.3, relax_b=0.4, eta=0.018, c=1.0, dx=1.0
    )
    reverse, _ = lane.photon_memory_log_shift(
        q[::-1],
        rho[::-1],
        drive=0.5,
        relax0=0.3,
        relax_b=0.4,
        eta=0.018,
        c=1.0,
        dx=1.0,
    )
    assert abs(forward - reverse) > 1e-6


def test_vq10_one_beta_ties_redshift_and_delay() -> None:
    q = np.array([0.2, 0.8, 0.4, 1.1], dtype=float)
    beta = 0.3
    h_ref = 0.01
    shift, delay = lane.slowed_ray_observables(q, eta=h_ref * beta, beta=beta, c=1.0, dx=1.0)
    i1 = lane.path_integral(q, 1.0)
    i2 = lane.path_integral(q**2, 1.0)
    assert delay == pytest.approx(beta * i1)
    assert -shift == pytest.approx(h_ref * beta * (i1 + beta * i2))
    zero_shift, zero_delay = lane.slowed_ray_observables(q, eta=0.0, beta=0.0, c=1.0, dx=1.0)
    assert zero_shift == zero_delay == 0.0


def test_report_has_exact_complete_matrix_and_all_gates_pass() -> None:
    config = raw_config()
    report = lane.synthetic_report(config)
    assert len(report["predictions"]) == 44
    assert {(row["fixture_id"], row["branch_id"]) for row in report["predictions"]} == {
        (fixture_id, branch_id)
        for fixture_id in lane.fixture_densities(config)
        for branch_id in lane._BRANCH_IDS
    }
    assert all(
        math.isfinite(row["log_frequency_shift"])
        and math.isfinite(row["extra_delay"])
        and math.isfinite(row["field_max"])
        for row in report["predictions"]
    )
    gates = lane.exact_gate_report(config)
    assert len(gates) == 10
    assert all(row["passed"] for row in gates)


def test_receipt_is_deterministic_claim_bounded_and_no_real_rows_are_opened() -> None:
    first, first_payloads = lane.build_receipt()
    second, second_payloads = lane.build_receipt()
    assert first == second
    assert first_payloads == second_payloads
    assert first["content_sha256"] == lane._self_hash(first)
    assert first["counts"] == {
        "branches": 11,
        "fixtures": 4,
        "predictions": 44,
        "operator_diagnostics": 4,
        "exact_gates": 10,
        "gates_passed": 10,
    }
    assert first["access_accounting"]["scientific_rows_decoded_by_this_packet"] == 0
    assert first["access_accounting"]["real_scores"] == 0
    assert first["claim_boundary"]["real_data_fit"] is False


def test_written_package_replays_from_only_the_canonical_paths() -> None:
    receipt = lane.check_package()
    assert receipt == lane.build_receipt()[0]
    assert Path(lane.OUTPUT_PATH).is_file()
