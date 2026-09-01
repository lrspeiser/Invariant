from __future__ import annotations

import copy
import inspect
import json
import math

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_same_law_eso325_extended_source_v5 as subject


def test_v4_is_byte_exact_preserved_as_blocked_evidence() -> None:
    config = subject.load_config()
    result = subject.verify_v4_preservation_and_sources(config)
    assert result["status"] == "PASS_V4_BYTE_EXACT_AND_V5_SOURCE_BINDINGS"
    assert len(result["v4_preserved_files"]) == 10
    assert any(
        row["path"] == "packet/receipt.json"
        and row["sha256"] == "5555b8e3e212ef569356b2b32b2a0a784e865e2227a306061c3b4fd8ee9e5c56"
        for row in result["v4_preserved_files"]
    )
    assert result["scientific_array_elements_decoded"] == 0
    assert result["scientific_response_values_opened"] == 0
    assert result["slacs_response_manifest_deserialized"] is False


def test_law_equations_metric_coefficients_constants_and_sources_are_exact_bound() -> None:
    config = subject.load_config()
    assert config["law_binding"] == subject.EXPECTED_LAW_BINDING
    assert config["law_sha256"] == subject.EXPECTED_LAW_SHA256
    assert config["source_binding"] == subject.EXPECTED_SOURCE_BINDING
    assert config["source_binding_sha256"] == subject.EXPECTED_SOURCE_SHA256
    assert subject._canonical_sha256(config["law_binding"]) == subject.EXPECTED_LAW_SHA256
    assert subject._canonical_sha256(config["source_binding"]) == subject.EXPECTED_SOURCE_SHA256
    assert "/100" not in json.dumps(config["law_binding"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["law_binding"]["field_equations"].__setitem__(
            "newton", "nabla^2 U = rho"
        ),
        lambda value: value["law_binding"].__setitem__("metric", "mutated"),
        lambda value: value["law_binding"]["coefficients"].__setitem__(
            "photon_integral_prefactor", 0.01
        ),
        lambda value: value["law_binding"]["constants"].__setitem__("c_km_s", 3.0e5),
        lambda value: value["law_binding"]["constants"].__setitem__(
            "G_kpc_km2_s2_Msun", 1.0
        ),
        lambda value: value["source_binding"][0].__setitem__("path", "other.fits"),
        lambda value: value["source_binding"][0].__setitem__("sha256", "0" * 64),
        lambda value: value["extended_density_contract"]["pseudo_nfw"].__setitem__(
            "density", "undefined"
        ),
        lambda value: value["predictive_likelihood_contract"].__setitem__(
            "posterior_predictive_mixture", "assigned score"
        ),
    ],
)
def test_in_memory_law_source_and_contract_mutations_fail_closed(mutation) -> None:
    forged = copy.deepcopy(subject.load_config())
    mutation(forged)
    with pytest.raises(subject.SameLawESO325V5Error):
        subject.validate_config(forged)


def test_physical_unit_ledger_and_planck_distances_close() -> None:
    config = subject.load_config()
    audit = subject.unit_audit(config)
    assert audit["all_pass"] is True
    assert audit["unexplained_numeric_photon_normalization"] is None
    assert all(row["left"] == row["right"] for row in audit["rows"])
    distances = subject.angular_diameter_distances(config)
    assert distances["D_l_Mpc"] == pytest.approx(148.4373468422846, rel=2e-14)
    assert distances["D_s_Mpc"] == pytest.approx(1760.104018743264, rel=2e-14)
    assert distances["D_ls_Mpc"] == pytest.approx(1710.545098103985, rel=2e-14)
    assert distances["D_ls_over_D_s"] == pytest.approx(0.9718431864756126, rel=2e-14)
    assert distances["lens_kpc_per_arcsec"] == pytest.approx(0.7196445653674095, rel=2e-14)


def test_photon_route_has_correct_explicit_physical_prefactor_and_no_law_knob() -> None:
    config = subject.load_config()
    coordinates, density, cell = subject.asymmetric_density(17, 8.0, 1.0e10)
    state = subject.solve_physical_state(
        density,
        coordinates,
        cell,
        g=0.18,
        range_kpc=3.0,
        padding_factor=2,
        config=config,
    )
    point = np.array([[1.0, 0.0]])
    observed = subject.reduced_photon_deflection(state, point, config)
    combined = state.Phi_km2_s2 + state.Psi_km2_s2
    gradient_x = np.gradient(combined, state.cell_kpc, edge_order=2)[0]
    integrated = np.sum(gradient_x, axis=2) * state.cell_kpc
    raw = subject.RegularGridInterpolator(
        (state.coordinates_kpc, state.coordinates_kpc), integrated, bounds_error=True
    )(point)[0]
    constants = config["law_binding"]["constants"]
    ratio = config["cosmology"]["frozen_distances"]["D_ls_over_D_s"]
    assert observed[0, 0] == pytest.approx(raw * ratio / constants["c_km_s"] ** 2)
    assert list(inspect.signature(subject.matter_acceleration).parameters) == [
        "state",
        "points_kpc",
    ]
    assert list(inspect.signature(subject.reduced_photon_deflection).parameters) == [
        "state",
        "lens_plane_points_kpc",
        "config",
    ]
    assert "g" not in inspect.signature(subject.reduced_photon_deflection).parameters
    assert "range_kpc" not in inspect.signature(subject.reduced_photon_deflection).parameters


def test_asymmetric_fixture_is_not_accidentally_symmetric() -> None:
    coordinates, density, cell = subject.asymmetric_density(33, 16.0, 1.0e11)
    assert coordinates[1] - coordinates[0] == pytest.approx(0.5)
    assert density.sum() * cell**3 == pytest.approx(1.0e11, rel=1e-12)
    for axis in range(3):
        assert subject._relative_rms(density, np.flip(density, axis=axis)) > 0.01


def test_all_reflections_permutations_real_padding_and_resolution_gates_pass() -> None:
    result = subject.target_free_gate(subject.load_config())
    metrics = result["metrics"]
    assert result["status"] == "PASS_PHYSICAL_UNIT_ASYMMETRIC_TARGET_FREE_GATES"
    assert result["pass"] is True
    assert set(metrics["all_three_reflection_relative_errors"]) == {
        "axis_0",
        "axis_1",
        "axis_2",
    }
    assert len(metrics["all_six_permutation_relative_errors"]) == math.factorial(3)
    assert max(metrics["all_three_reflection_relative_errors"].values()) <= 5e-10
    assert max(metrics["all_six_permutation_relative_errors"].values()) <= 5e-10
    assert result["grid"]["primary_padding_factor"] == 2
    assert result["grid"]["comparison_padding_factor"] == 4
    assert result["grid"]["cell_ratio"] == pytest.approx(2.0)
    assert metrics["doubled_padding_observable_relative_rms"] <= 0.03
    assert metrics["halved_cell_observable_relative_rms"] <= 0.08


def test_candidate_residual_and_holdout_lpd_are_calculated_not_assigned() -> None:
    result = subject.target_free_gate(subject.load_config())
    metrics = result["metrics"]
    assert 0.0 < metrics["candidate_chi2_per_datum_calculated"] < 1.0
    assert metrics["candidate_chi2_per_datum_calculated"] < metrics["gr_chi2_per_datum_calculated"]
    assert metrics["candidate_minus_gr_holdout_lpd"] == pytest.approx(
        metrics["candidate_holdout_log_predictive_density_calculated"]
        - metrics["gr_holdout_log_predictive_density_calculated"]
    )
    assert metrics["candidate_minus_gr_holdout_lpd"] >= 10.0
    assert metrics["recovered_g"] == pytest.approx(0.18, abs=0.01)
    assert set(metrics["channel_weighted_residuals"]) == {
        "matter",
        "lensing",
        "extended_image",
    }


def test_gaussian_log_predictive_density_includes_normalization() -> None:
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([0.5, 2.0, 2.0])
    sigma = np.array([0.5, 1.0, 2.0])
    indices = np.array([0, 2])
    expected = sum(
        -0.5 * ((observed[index] - predicted[index]) / sigma[index]) ** 2
        - math.log(sigma[index])
        - 0.5 * math.log(2.0 * math.pi)
        for index in indices
    )
    assert subject.gaussian_log_predictive_density(
        observed, predicted, sigma, indices
    ) == pytest.approx(expected)


def test_pseudo_nfw_and_scientific_predictive_metric_are_fully_specified() -> None:
    config = subject.load_config()
    pseudo = config["extended_density_contract"]["pseudo_nfw"]
    assert "rho_s*r_s^3" in pseudo["density"]
    assert "20*r_s" in pseudo["truncation"] and "22*r_s" in pseudo["truncation"]
    assert "f_DM/(1-f_DM)" in pseudo["normalization"]
    predictive = config["predictive_likelihood_contract"]
    assert predictive["metric_not_evidence"].startswith("No marginal evidence")
    assert "logsumexp" in predictive["posterior_predictive_mixture"]
    assert "-log(4096)" in predictive["posterior_predictive_mixture"]
    assert "C_hh-C_ht*C_tt^-1*C_th" in predictive["muse_predictive_density"]
    assert "sector=floor" in predictive["sector_assignment"]


def test_external_missing_inputs_are_separate_from_post_decode_empirical_gates() -> None:
    result = subject.source_readiness(subject.load_config())
    assert result["status"] == "SOURCE_BLOCKED_MISSING_EXTERNAL_TEMPLATE_GAIA_AND_EXTRACTOR_BINDINGS"
    assert result["external_missing_count"] == 4
    assert result["post_decode_gate_count"] == 5
    assert all("field star" not in item.lower() for item in result["missing_external_inputs"])
    assert any("PSF stars" in item for item in result["deferred_empirical_gates"])
    assert any("covariance" in item for item in result["deferred_empirical_gates"])
    assert result["scientific_array_elements_decoded"] == 0
    assert result["slacs_response_values_opened"] == 0


def test_artifacts_are_deterministic_and_claims_remain_source_blocked() -> None:
    config = subject.load_config()
    first = subject.build_artifacts(config)
    second = subject.build_artifacts(config)
    assert first == second
    assert set(first) == {
        "v4-preservation-and-exact-source-receipt.json",
        "physical-unit-and-distance-audit.json",
        "target-free-physical-shared-state-gate.json",
        "source-readiness-split.json",
        "frozen-density-and-predictive-contract.json",
        "report.md",
    }
    assert b"no arbitrary photon normalization" in first["report.md"]
    assert b"Scientific FITS arrays decoded: **0**" in first["report.md"]


def test_receipt_is_physical_target_free_pass_and_empirical_source_block() -> None:
    receipt = subject.build_receipt()
    assert receipt["status"] == "SOURCE_BLOCKED_EXTERNAL_INPUTS_AFTER_PHYSICAL_TARGET_FREE_PASS"
    assert receipt["decision"] == "NO_ESO_ARRAY_DECODE_NO_ESO_SCORE_KEEP_SLACS_SEALED"
    assert receipt["target_free_gate"]["pass"] is True
    assert receipt["access_accounting"]["scientific_fits_array_elements_decoded"] == 0
    assert receipt["access_accounting"]["eso_response_values_opened"] == 0
    assert receipt["access_accounting"]["eso_scores_computed"] == 0
    assert receipt["access_accounting"]["slacs_response_values_opened"] == 0


def test_packet_matches_deterministic_rebuild_if_present() -> None:
    if not subject.OUTPUT_PATH.exists():
        pytest.skip("packet is sealed after unit tests")
    assert json.loads(subject.OUTPUT_PATH.read_text(encoding="utf-8")) == subject.build_receipt()
    for name, payload in subject.build_artifacts(subject.load_config()).items():
        assert (subject.ARTIFACT_DIRECTORY / name).read_bytes() == payload
