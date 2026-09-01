from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_lane6_gqns_solar_source_domain_v1 as lane


@lru_cache(maxsize=1)
def _built() -> tuple[dict, dict[str, bytes]]:
    return lane.build_receipt()


def test_exact_package_pins_and_append_only_predecessor() -> None:
    config = lane.load_config()
    assert lane.file_sha256(lane._repo_path(lane.CONFIG_PATH)) == lane._CONFIG_RAW_SHA256
    assert lane.content_sha256(config) == lane._CONFIG_CONTENT_SHA256
    assert lane.module_semantic_sha256() == lane._MODULE_SEMANTIC_SHA256
    assert lane.file_sha256(lane._repo_path(lane.TEST_PATH)) == lane._TEST_RAW_SHA256
    observed = lane.validate_predecessor(config)
    assert set(observed) == {"config", "module", "test", "receipt"}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "OPEN"),
        (("unchanged_gqns_law", "retuned"), True),
        (("unchanged_gqns_law", "parameters_fit"), 1),
        (("time_grid", "samples"), 12),
        (("access_contract", "observational_response_rows_opened"), 1),
        (("outputs", "receipt"), "wrong.json"),
    ],
)
def test_semantic_mutations_reject(path: tuple[str, ...], value: object) -> None:
    config = lane.load_config(verify_package=False)
    mutated = copy.deepcopy(config)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(lane.GQNSSolarError):
        lane.validate_config(mutated)


def test_jpl_table1_j2000_emb_coordinate_regression() -> None:
    config = lane.load_config()
    position = lane.planet_position(config["jpl_table1_elements"]["EMB"], 0.0)
    assert position == pytest.approx(
        [-0.177171249104624, 0.967214484966116, -2.58449294018e-7], rel=2.0e-12
    )
    assert np.linalg.norm(position) == pytest.approx(0.983307434854, rel=2.0e-11)


def test_geometry_dimensions_sphere_shutoff_and_oblateness_activation() -> None:
    config = lane.load_config()
    domains = {row["id"]: row for row in config["source_domains"]}
    sphere = lane.geometry_metrics(lane._domain_bodies(config, domains["D00_SUN_SPHERE_ONLY"], 0.0))
    oblate = lane.geometry_metrics(lane._domain_bodies(config, domains["D01_SUN_OBLATE_ONLY"], 0.0))
    assert sphere["A_Q"] < 1.0e-15
    assert sphere["L_au"] > 0.0
    assert 0.0 < oblate["A_Q"] < 1.0e-6
    assert sum(oblate["eigenvalues_au2"]) == pytest.approx(oblate["L_au"] ** 2)


def test_normalized_helmholtz_kernel_enclosed_mass_and_limits() -> None:
    assert lane.yukawa_enclosed_fraction(0.0) == 0.0
    assert lane.yukawa_enclosed_fraction(1.0e-7) == pytest.approx(0.5e-14, rel=1.0e-7)
    assert lane.yukawa_enclosed_fraction(1.0) == pytest.approx(1.0 - 2.0 / np.e)
    assert lane.yukawa_enclosed_fraction(100.0) == pytest.approx(1.0)
    grid = np.linspace(0.001, 20.0, 1000)
    values = np.asarray([lane.yukawa_enclosed_fraction(float(value)) for value in grid])
    assert np.all(np.diff(values) > 0.0)


def test_arbitrary_so3_translation_and_colocated_split_controls() -> None:
    receipt, payloads = _built()
    sensitivity = __import__("json").loads(payloads["source-domain-sensitivity.json"])
    rotation = sensitivity["arbitrary_SO3"]
    assert rotation["rotations"] == 64
    assert rotation["max_A_Q_absolute_error"] < 2.0e-15
    assert rotation["max_L_au_absolute_error"] < 2.0e-15
    assert rotation["max_force_covariance_error_m_s2"] < 1.0e-16
    assert sensitivity["translation"]["A_Q_absolute_error"] < 2.0e-14
    assert sensitivity["translation"]["L_au_absolute_error"] < 2.0e-14
    assert sensitivity["co_located_mass_split"]["A_Q_absolute_error"] < 2.0e-15
    assert sensitivity["co_located_mass_split"]["L_au_absolute_error"] < 2.0e-15
    assert receipt["summary"]["arbitrary_SO3_max_force_error_m_s2"] < 1.0e-16


def test_named_source_boundaries_change_the_global_functional() -> None:
    _, payloads = _built()
    sensitivity = __import__("json").loads(payloads["source-domain-sensitivity.json"])
    named = sensitivity["named_boundary_J2000"]
    assert named["D00_SUN_SPHERE_ONLY"]["A_Q"] < 1.0e-15
    assert named["D05_SUN_EIGHT_PLANETS"]["A_Q"] > 0.1
    assert named["D05_SUN_EIGHT_PLANETS"]["L_au"] > 0.1
    assert len({round(row["A_Q"], 8) for row in named.values()}) >= 5
    assert len({round(row["L_au"], 8) for row in named.values()}) >= 5


def test_decomposition_is_nonlinear_and_remote_sources_change_moments() -> None:
    _, payloads = _built()
    sensitivity = __import__("json").loads(payloads["source-domain-sensitivity.json"])
    for target in ("EARTH", "NEPTUNE"):
        assert sensitivity["nonlinear_decomposition"][target]["relative_difference"] > 0.01
    remote = sensitivity["remote_source_boundary"]
    assert remote[1]["L_au"] > remote[0]["L_au"] * 100.0
    assert remote[1]["A_Q"] > 0.9


def test_moon_asteroid_and_oblateness_sensitivities_are_retained() -> None:
    _, payloads = _built()
    sensitivity = __import__("json").loads(payloads["source-domain-sensitivity.json"])
    assert sensitivity["moon_sensitivity"]["A_Q_delta"] != 0.0
    assert sensitivity["asteroid_sensitivity"]["A_Q_delta"] != 0.0
    solar = sensitivity["solar_oblateness_sensitivity"]
    assert solar["A_Q_sphere"] < 1.0e-15
    assert solar["A_Q_J2"] > 0.0


def test_rho_dark_and_force_implications_are_finite_and_positive() -> None:
    _, payloads = _built()
    sensitivity = __import__("json").loads(payloads["source-domain-sensitivity.json"])
    density = sensitivity["D05_J2000_sun_cloud"]
    assert all(row["sun_cloud_density_kg_m3"] >= 0.0 for row in density)
    assert all(0.0 < row["sun_cloud_enclosed_mass_over_sun_mass"] < 1.0 for row in density)
    assert density[0]["sun_cloud_density_kg_m3"] > density[-1]["sun_cloud_density_kg_m3"]
    assert (
        density[0]["sun_cloud_enclosed_mass_over_sun_mass"]
        < density[-1]["sun_cloud_enclosed_mass_over_sun_mass"]
    )


def test_time_and_force_inventory_is_complete() -> None:
    receipt, _ = _built()
    assert receipt["summary"]["source_domains"] == 8
    assert receipt["summary"]["time_samples"] == 601
    assert receipt["summary"]["domain_time_rows"] == 8 * 601
    assert receipt["summary"]["force_summary_rows"] == 8 * 8
    assert receipt["summary"]["targets"] == 8


def test_common_and_per_planet_inverse_square_nuisances_are_reported() -> None:
    _, payloads = _built()
    text = payloads["force-summary.csv"].decode()
    assert "common_inverse_square_scale" in text
    assert "per_target_inverse_square_scale" in text
    assert "common_scale_residual_max_m_s2" in text
    assert "per_target_scale_residual_max_m_s2" in text


def test_bounded_negative_theorem_and_large_published_bound_margin() -> None:
    receipt, payloads = _built()
    theorem = __import__("json").loads(payloads["bounded-negative-theorem.json"])
    assert theorem["decision"] == "DECISIVELY_EXCLUDED_AS_UNCHANGED_GLOBAL_SOLAR_SOURCE_LAW"
    assert theorem["D05_median_A_Q"] > 0.1
    assert theorem["minimum_pairwise_fractional_spread_before_a_common_rescale"] > 0.01
    assert theorem["D05_Neptune_common_scale_residual_to_published_outer_bound"] > 1000.0
    assert receipt["decision"] == theorem["decision"]


def test_host_only_sun_is_not_silently_called_a_repair() -> None:
    receipt, payloads = _built()
    theorem = __import__("json").loads(payloads["bounded-negative-theorem.json"])
    assert "new covariant source-localization" in theorem["localization_boundary"]
    assert receipt["retained_failures"]["host_only_localization_is_new_theory"] is True


def test_access_claim_boundaries_and_failures_are_preserved() -> None:
    receipt, _ = _built()
    access = receipt["access_contract"]
    assert access["ephemeris_binary_files_downloaded"] == 0
    assert access["observational_response_files_opened"] == 0
    assert access["observational_response_rows_opened"] == 0
    assert access["parameters_fit_to_responses"] == 0
    assert access["model_calls"] == 0
    assert access["paid_calls"] == 0
    assert receipt["retained_failures"]["predecessor_counterexamples_preserved"] is True
    assert "a precision DE440 or INPOP refit" in receipt["claim_boundary"]["does_not_establish"]


def test_deterministic_artifacts_hashes_and_report_boundaries() -> None:
    first, first_payloads = _built()
    second, second_payloads = lane.build_receipt()
    assert first == second
    assert first_payloads == second_payloads
    assert len(first["artifact_index"]) == 6
    for row in first["artifact_index"]:
        name = lane.Path(row["path"]).name
        assert lane.hashlib.sha256(first_payloads[name]).hexdigest() == row["sha256"]
    report = first_payloads["report.md"].decode()
    assert "DECISIVELY_EXCLUDED" in report
    assert "not a precision DE440 or INPOP refit" in report
    assert "new source-localization law" in report
