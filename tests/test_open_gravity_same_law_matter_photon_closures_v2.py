from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_same_law_matter_photon_closures_v2 as subject


def test_v1_receipt_is_pinned_and_explicitly_blocked() -> None:
    config = subject.load_config()
    prior = config["supersedes"]
    assert subject.file_sha256(Path(prior["path"])) == prior["file_sha256"]
    assert prior["audit_status"].startswith("BLOCKED_")


def test_one_executable_state_constructs_both_metric_potentials() -> None:
    config = subject.load_config()
    state = subject.field_state(0.02, config)
    parameters = config["parameter_set"]
    coupling = parameters["universal_coupling_g"]
    assert state["Phi"] == pytest.approx(state["U"] + 4.0 * coupling * state["Y"] / 3.0)
    assert state["Psi"] == pytest.approx(state["U"] + 2.0 * coupling * state["Y"] / 3.0)
    assert state["g_tt"] < 0.0
    assert state["g_space"] > 0.0


def test_selected_state_satisfies_exterior_field_equation() -> None:
    residuals = subject.field_equation_residuals(subject.load_config())
    assert residuals["equation"].endswith("outside r=0")
    assert max(abs(row["relative_residual"]) for row in residuals["rows"]) < 2.0e-5


def test_timelike_motion_comes_from_metric_not_assigned_acceleration() -> None:
    config = subject.load_config()
    radius = config["parameter_set"]["dynamics_radii"][1]
    acceleration = subject.timelike_acceleration(radius, config)
    phi_prime = subject.state_derivatives(radius, config)["Phi_prime"]
    assert acceleration == pytest.approx(-phi_prime, rel=2.0e-8)


def test_independent_dynamics_and_ray_routes_recover_fixed_slip() -> None:
    result = subject.independent_route_slip(subject.load_config())
    assert len(result["dynamics_rows"]) == 4
    assert len(result["lensing_rows"]) == 4
    assert result["used_stored_phi_or_psi_extra"] is False
    assert result["used_tautological_two_g_lens_minus_g_dyn"] is False
    assert result["fitted_phi_extra_amplitude"] == pytest.approx(0.4, rel=2.0e-5)
    assert result["fitted_phi_plus_psi_extra_amplitude"] == pytest.approx(0.6, rel=2.0e-5)
    assert result["reconstructed_gamma_extra"] == pytest.approx(0.5, abs=2.0e-5)


def test_null_observables_are_derived_from_same_metric() -> None:
    config = subject.load_config()
    result = subject.derived_observables(config)
    assert result["photon_only_parameters"] == 0
    assert 0.0 < result["photon_coordinate_speed"] < config["parameter_set"]["c"]
    assert result["deflection"] > 0.0
    assert result["shapiro_delay"] > 0.0
    assert 0.0 < result["gravitational_frequency_ratio"] < 1.0
    assert result["unsupported_channels"] == []


def test_image_positions_and_delay_use_no_separate_fermat_knob() -> None:
    result = subject.image_delay_fixture(subject.load_config())
    assert result["negative_image"] < 0.0 < result["positive_image"]
    assert result["signed_delay"] != 0.0
    assert result["separate_fermat_coefficient"] == 0.0


def test_metric_propagation_is_achromatic_reciprocal_and_distance_dual() -> None:
    result = subject.derived_observables(subject.load_config())
    assert result["chromaticity"]["difference"] == 0.0
    assert result["chromaticity"]["metric_is_frequency_independent"] is True
    assert result["distance_duality"]["eta"] == pytest.approx(1.0, abs=1e-14)
    assert result["distance_duality"]["photon_number_survival_fraction"] == 1.0
    assert result["reciprocity"]["deflection_reversal_error"] == pytest.approx(0.0, abs=1e-14)
    assert result["reciprocity"]["shapiro_path_reversal_error"] == pytest.approx(0.0, abs=1e-14)


def test_tensor_cone_and_group_speed_use_the_same_mediator_mass() -> None:
    config = subject.load_config()
    result = subject.tensor_propagation(config)
    assert result["mediator_mass_used"] == config["parameter_set"]["mediator_mass_mu"]
    assert result["characteristic_speed_over_c"] == 1.0
    assert 0.0 < result["group_speed_over_c"] < 1.0


def test_frozen_ray_grid_converges_when_doubled() -> None:
    errors = subject.quadrature_convergence(subject.load_config())["relative_errors"]
    assert errors["deflection"] < 5.0e-4
    assert errors["shapiro_delay"] < 1.0e-5
    assert errors["image_delay"] < 3.0e-3
    assert errors["slip"] < 1.0e-10


def test_same_law_parameters_contain_no_photon_opacity_or_path_knob() -> None:
    config = subject.load_config()
    serialized = json.dumps(config["parameter_set"], sort_keys=True).lower()
    for forbidden in ("photon", "opacity", "dispersion", "path", "fermat"):
        assert forbidden not in serialized
    assert tuple(config["same_law_gate"]["forbidden"]) == subject._FORBIDDEN


def test_exact_eso_hst_and_muse_metadata_are_frozen_but_source_blocked() -> None:
    manifest = subject.load_config()["eso325_source_manifest"]
    assert manifest["status"].startswith("SOURCE_BLOCKED_")
    assert [row["product_filename"] for row in manifest["hst_products"]] == [
        "hst_10429_09_acs_wfc_f814w_j95t09_drc.fits",
        "hst_10429_10_acs_wfc_f475w_j95t10_drc.fits",
    ]
    assert [row["bytes"] for row in manifest["hst_products"]] == [369486720, 367663680]
    assert manifest["muse_products"][0]["dp_id"] == "ADP.2016-09-07T12:23:32.515"
    assert manifest["muse_products"][0]["bytes"] == 7378352640
    assert "Student-t(nu=4)" in manifest["likelihood"]["joint_log_likelihood"]
    assert "SHA256 for each payload" in manifest["unresolved_before_open"]


def test_slacs_holdout_is_the_unchanged_preexisting_sealed_split() -> None:
    config = subject.load_config()
    manifest = config["slacs_holdout_manifest"]
    assert manifest["status"] == "UNCHANGED_CONFIRMATION_SEALED"
    assert manifest["eligible"] == 57
    assert manifest["exploration"] == 45
    assert manifest["reserved_confirmation"] == 12
    assert manifest["confirmation_opened"] is False
    assert (
        subject.file_sha256(Path(manifest["sample_manifest_path"]))
        == manifest["sample_manifest_sha256"]
    )


def test_all_mandatory_countermodels_are_retained() -> None:
    ids = {row["id"] for row in subject.load_config()["countermodels"]}
    assert ids == {
        "CM01_GR_STARS_NFW",
        "CM02_MASS_SHEET_SOURCE_POSITION",
        "CM03_STELLAR_ANISOTROPY_INCLINATION",
        "CM04_LINE_OF_SIGHT_EXTERNAL_CONVERGENCE",
        "CM05_PLASMA_DUST_SCATTERING",
        "CM06_MOVING_LENS_AND_SOURCE_VARIABILITY",
        "CM07_TEVES_DISFORMAL",
        "CM08_COVARIANT_NONLOCAL_GRAVITY",
        "CM09_CAUSAL_METRIC_MEMORY",
        "CM10_NONMETRIC_PATH_MEMORY",
    }


def test_adversarial_photon_source_and_confirmation_mutations_fail_closed() -> None:
    config = subject.load_config()
    mutations = (
        lambda value: value["parameter_set"].__setitem__("photon_multiplier", 1.0),
        lambda value: value["same_law_gate"].__setitem__("all_parameters_shared", False),
        lambda value: value["eso325_source_manifest"].__setitem__("status", "READY"),
        lambda value: value["slacs_holdout_manifest"].__setitem__("confirmation_opened", True),
        lambda value: value["access_contract"].__setitem__("scientific_response_rows_opened", 1),
        lambda value: value["countermodels"].pop(),
    )
    for mutation in mutations:
        forged = copy.deepcopy(config)
        mutation(forged)
        with pytest.raises(subject.SameLawV2Error):
            subject.validate_config(forged)


def test_receipt_has_split_pass_block_and_requests_reaudit() -> None:
    receipt = subject.build_receipt()
    assert receipt["status"] == "PASS_EXECUTABLE_SAME_LAW_BLOCK_REAL_SOURCES"
    assert receipt["same_law_pass"] is True
    assert receipt["photon_only_parameters"] == 0
    assert receipt["blocked_channels"] == []
    assert receipt["source_status"]["scientific_response_rows_opened"] == 0
    assert receipt["decision"] == (
        "REQUEST_INDEPENDENT_REAUDIT_BEFORE_ANY_ESO_OR_SLACS_RESPONSE_ACCESS"
    )


def test_artifacts_are_deterministic_and_source_bounded() -> None:
    config = subject.load_config()
    first = subject.build_artifacts(config)
    second = subject.build_artifacts(config)
    assert first == second
    assert set(first) == {
        "field-equation-residuals.json",
        "derived-observables.json",
        "independent-route-slip.json",
        "numerical-convergence.json",
        "countermodels.json",
        "source-and-likelihood-manifests.json",
        "report.md",
    }
    assert b"Theory execution PASS; observational sources BLOCKED" in first["report.md"]


def test_atomic_no_clobber(tmp_path) -> None:
    path = tmp_path / "sealed.bin"
    assert subject._atomic_no_clobber(path, b"one") == "CREATED"
    assert subject._atomic_no_clobber(path, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(subject.SameLawV2Error, match="existing artifact differs"):
        subject._atomic_no_clobber(path, b"two")
