from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import pytest
from scipy.integrate import quad

from sigma_theory_compiler import open_gravity_path_accumulated_weyl_redshift_v1 as pathlaw


def test_dimensions_and_exact_controls_are_frozen() -> None:
    config = pathlaw.load_config()
    synthetic = pathlaw.synthetic_benchmarks(config)
    assert synthetic["dimensions"] == {
        "d_ell": "length",
        "path_age_s": "length",
        "hubble_length_L_H": "length",
        "weyl_driver_q_W": "length^-1",
        "activation_A": "1",
        "exposure_E_gamma": "1",
        "coupling_alpha": "1",
        "extra_log_redshift": "1",
    }
    assert len(config["exact_controls"]) == 8
    assert all(synthetic["checks"].values())


def test_exact_gr_conformal_and_endpoint_limits() -> None:
    assert 0.0 * pathlaw.point_mass_exposure_analytic(1.0, 1.0) == 0.0
    assert pathlaw.point_mass_exposure_analytic(1.0, 0.0) == 0.0
    assert pathlaw.exact_one_form_integral(-0.2, 0.7) == pytest.approx(0.9)
    assert pathlaw.exact_one_form_integral(-0.2, 0.7) == pathlaw.exact_one_form_integral(-0.2, 0.7)


def test_piecewise_integral_is_invariant_to_segment_subdivision() -> None:
    coarse = pathlaw.integrate_piecewise_exposure(
        [0.2, 0.3, 0.5], [0.4, 0.4, 0.4], hubble_length=2.0, initial_path_age=0.7
    )
    fine = pathlaw.integrate_piecewise_exposure(
        [0.1, 0.1, 0.1, 0.2, 0.25, 0.25],
        [0.4] * 6,
        hubble_length=2.0,
        initial_path_age=0.7,
    )
    assert coarse == pytest.approx(fine, rel=0.0, abs=2.0e-16)


def test_point_mass_analytic_law_matches_independent_infinite_quadrature() -> None:
    for impact in (2.0, 5.0, 10.0):
        analytic = pathlaw.point_mass_exposure_analytic(impact, 4.785e-6)
        numeric = pathlaw.point_mass_exposure_numeric(impact, 4.785e-6)
        assert numeric == pytest.approx(analytic, rel=1.0e-10)
    assert pathlaw.point_mass_exposure_analytic(
        2.0, 4.785e-6
    ) > pathlaw.point_mass_exposure_analytic(10.0, 4.785e-6)


def test_exact_point_mass_lens_geometry_and_weyl_exposure_are_consistent() -> None:
    for flux_ratio in (0.01, 0.14, 0.25, 0.5, 0.9, 1.0):
        geometry = pathlaw.point_mass_angular_geometry(flux_ratio, 2.4)
        source_offset = geometry["source_offset_over_einstein_radius"]
        outer_root = geometry["signed_x_plus"]
        inner_root = geometry["signed_x_minus"]
        assert outer_root - 1.0 / outer_root == pytest.approx(source_offset, abs=2.0e-15)
        assert inner_root - 1.0 / inner_root == pytest.approx(source_offset, abs=2.0e-15)
        assert outer_root * inner_root == pytest.approx(-1.0, abs=2.0e-15)
        assert geometry["reconstructed_separation_arcsec"] == pytest.approx(2.4, abs=2.0e-15)
        assert geometry["reconstructed_flux_ratio"] == pytest.approx(flux_ratio, rel=3.0e-14)

        outer_exposure = pathlaw.point_mass_exposure_analytic(geometry["outer_impact_arcsec"], 1.0)
        inner_exposure = pathlaw.point_mass_exposure_analytic(geometry["inner_impact_arcsec"], 1.0)
        assert inner_exposure / outer_exposure == pytest.approx(
            math.sqrt(geometry["outer_impact_arcsec"] / geometry["inner_impact_arcsec"]),
            rel=2.0e-15,
        )


def test_path_age_activation_is_causal_and_solar_suppressed() -> None:
    assert pathlaw.activation(0.0, 10.0) == 0.0
    assert 0.0 < pathlaw.activation(1.0, 10.0) < pathlaw.activation(2.0, 10.0) < 1.0
    synthetic = pathlaw.synthetic_benchmarks(pathlaw.load_config())
    assert synthetic["solar"]["gated_exposure_upper_bound"] < 1.0e-16
    with pytest.raises(pathlaw.PathAccumulationError):
        pathlaw.activation(-1.0, 10.0)


def test_flrw_baryon_frame_path_age_is_the_invariant_light_travel_integral() -> None:
    config = pathlaw.load_config()
    cosmology = config["real_preflight"]["cosmology"]
    h0 = float(cosmology["H0_km_s_Mpc"])
    omega_m = float(cosmology["omega_m"])
    omega_lambda = float(cosmology["omega_lambda"])
    c_km_s = float(cosmology["c_km_s"])
    for z_near, z_far in ((0.1, 0.3), (0.5, 1.5), (1.0, 3.0)):
        independent, _ = quad(
            lambda z: 1.0 / ((1.0 + z) * math.sqrt(omega_m * (1.0 + z) ** 3 + omega_lambda)),
            z_near,
            z_far,
            epsabs=1.0e-13,
            epsrel=1.0e-13,
        )
        expected = c_km_s * independent / h0
        observed = pathlaw.baryon_frame_path_length_mpc(z_near, z_far, cosmology)
        assert observed == pytest.approx(expected, rel=1.0e-14)


def test_response_blind_lens_preflight_uses_exactly_eight_exploration_rows() -> None:
    rows = pathlaw.lens_prediction_rows(pathlaw.load_config())
    assert len(rows) == 8
    assert len({row["name"] for row in rows}) == 8
    assert {row["fold"] for row in rows} == {0, 1, 2, 3}
    assert all(row["response_opened"] is False for row in rows)
    assert all(row["source_model"] == "MODEL_LIFTED_EXACT_POINT_MASS" for row in rows)
    assert all(row["delta_velocity_km_s_per_alpha"] > 0.0 for row in rows)
    assert all(row["inner_impact_arcsec"] < row["outer_impact_arcsec"] for row in rows)
    assert all(row["inner_impact_mpc"] < row["outer_impact_mpc"] for row in rows)
    assert all(row["model_gravitational_radius_mpc"] > 0.0 for row in rows)
    assert max(row["path_measure_quadrature_relative_error"] for row in rows) < 1.0e-13
    assert all(
        row["source_to_lens_path_mpc"] < row["rejected_scaled_comoving_proxy_mpc"] for row in rows
    )
    assert min(row["rejected_proxy_relative_difference"] for row in rows) > 0.1
    assert all(
        row["inner_impact_mpc"] / row["outer_impact_mpc"]
        == pytest.approx(
            row["inner_impact_arcsec"] / row["outer_impact_arcsec"],
            rel=2.0e-15,
        )
        for row in rows
    )
    assert max(row["maximum_lens_equation_residual"] for row in rows) < 1.0e-14
    assert max(row["exposure_scaling_absolute_error"] for row in rows) < 1.0e-14


def test_actual_source_row_access_is_exact_and_confirmation_predictors_are_unused() -> None:
    config = pathlaw.load_config()
    access = pathlaw.lens_source_access(config)
    accounting = access["accounting"]
    assert accounting["source_predictor_rows_parsed"] == 12
    assert accounting["exploration_predictor_rows_used"] == 8
    assert accounting["confirmation_predictor_rows_parsed"] == 4
    assert accounting["confirmation_predictor_rows_used"] == 0
    assert accounting["confirmation_response_rows_opened"] == 0

    paths = {row["role"]: row["path"] for row in config["bindings"]}
    with Path(paths["LENS_PREDICTORS"]).open(encoding="utf-8", newline="") as source_handle:
        source_rows = list(csv.DictReader(source_handle, delimiter="\t"))
    manifest = json.loads(Path(paths["FROZEN_SAMPLE_MANIFEST"]).read_text(encoding="utf-8"))
    original = pathlaw._select_exploration_rows(source_rows, manifest)

    confirmation_names = {
        row["identity"]
        for row in manifest["objects"]
        if row.get("lane") == "photon_delay" and row.get("role") == "confirmation"
    }
    adversarial = copy.deepcopy(source_rows)
    for row in adversarial:
        if row["name"] in confirmation_names:
            row["image_flux_ratio"] = "not-a-number-and-must-not-be-consumed"
            row["image_separation_arcsec"] = "not-a-number-and-must-not-be-consumed"
    attacked = pathlaw._select_exploration_rows(adversarial, manifest)
    assert attacked["selected"] == original["selected"]
    assert attacked["accounting"] == original["accounting"]


def test_receipt_retains_conservation_and_response_blocks() -> None:
    receipt = pathlaw.build_receipt()
    assert receipt["law"]["coordinate_invariant"]
    assert receipt["law"]["ray_reparameterization_invariant"]
    assert receipt["conservation"] == {
        "photon_number_can_be_conserved": True,
        "photon_stress_energy_conserved_for_nonzero_alpha_in_stationary_lens": False,
        "compensating_action_derived": False,
        "status": "BLOCK_ACTION_AND_COMPENSATING_STRESS_ENERGY_NOT_DERIVED",
    }
    assert receipt["real_preflight"]["raw_response_rows_opened"] == 0
    assert receipt["real_preflight"]["confirmation_response_rows_opened"] == 0
    assert receipt["real_preflight"]["confirmation_predictor_rows_parsed"] == 4
    assert receipt["real_preflight"]["confirmation_predictor_rows_used"] == 0
    assert all(receipt["real_preflight"]["checks"].values())
    assert receipt["publication_ready"] is False
    assert "BLOCK_DYNAMICAL_COMPLETION" in receipt["decision"]


def test_comparators_and_primary_papers_prevent_false_novelty_claim() -> None:
    config = pathlaw.load_config()
    comparators = {row["comparator"] for row in config["comparator_map"]}
    assert comparators == {
        "GR stationary endpoint redshift",
        "GR integrated Sachs-Wolfe/Rees-Sciama",
        "cosmological redshift",
        "plasma",
        "dust",
        "static lensing and time delay",
        "moving-lens and differential-magnification frequency shifts",
    }
    roles = {row["role"] for row in config["primary_literature"]}
    assert {
        "GR_PATH_EFFECT_CONTROL",
        "NONINTEGRABLE_WEYL_PREDECESSOR",
        "TIRED_LIGHT_PREDECESSOR",
        "MOVING_LENS_COMPARATOR",
        "MULTI_IMAGE_SPECTROSCOPY_PRIMARY",
    } <= roles
    assert config["novelty_disposition"]["claim"] == (
        "POTENTIALLY_NEW_TESTABLE_SYNTHESIS_NOT_HISTORICAL_NOVELTY"
    )


def test_config_mutation_fails_closed() -> None:
    config = pathlaw.load_config()
    mutated = copy.deepcopy(config)
    mutated["law"]["free_parameters"].append({"symbol": "beta"})
    with pytest.raises(pathlaw.PathAccumulationError):
        pathlaw.validate_config(mutated)


def test_input_bindings_are_source_or_metadata_only() -> None:
    config = pathlaw.load_config()
    bindings = pathlaw.validate_input_bindings(config)
    assert set(bindings) == {
        "LENS_PREDICTORS",
        "FROZEN_SAMPLE_MANIFEST",
        "PREDICTOR_SOURCE_MANIFEST",
        "TEMPORAL_IDENTIFIABILITY_RECEIPT",
        "PRIMARY_SOURCE_COMPARATOR_CONTRACT",
        "BLOCKED_INVARIANT_FLRW_PATH_AGE_RECEIPT",
    }
    assert not any("response" in row["path"].lower() for row in config["bindings"])
    assert config["access_contract"]["raw_response_files_opened"] == 0
    assert config["access_contract"]["confirmation_predictor_rows_parsed"] == 4


def test_receipt_self_hash_detects_forgery() -> None:
    receipt = pathlaw.build_receipt()
    pathlaw.validate_receipt_content(receipt)
    forged = copy.deepcopy(receipt)
    forged["publication_ready"] = True
    with pytest.raises(pathlaw.PathAccumulationError):
        pathlaw.validate_receipt_content(forged)


def test_atomic_package_round_trip_is_deterministic(tmp_path) -> None:
    output = tmp_path / "path-law"
    assert pathlaw.write_package(output) == "CREATED"
    pathlaw.validate_package(output)
    assert pathlaw.write_package(output) == "EXISTING_IDENTICAL"
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert len(receipt["artifacts"]) == 3
    prediction = output / "artifacts/exploration-lens-predictions.csv"
    assert prediction.read_text(encoding="utf-8").count("\n") == 9


def test_path_law_rejects_negative_driver_or_noncausal_age() -> None:
    with pytest.raises(pathlaw.PathAccumulationError):
        pathlaw.integrate_piecewise_exposure([1.0], [-1.0], hubble_length=10.0)
    with pytest.raises(pathlaw.PathAccumulationError):
        pathlaw.integrate_piecewise_exposure(
            [1.0], [1.0], hubble_length=10.0, initial_path_age=-1.0
        )
    assert math.isfinite(pathlaw.integrate_piecewise_exposure([1.0], [1.0], hubble_length=10.0))
