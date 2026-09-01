from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_same_law_matter_photon_closures_v3 as subject


def test_v2_is_pinned_as_blocked_append_only_evidence() -> None:
    config = subject.load_config()
    prior = config["supersedes"]
    path = Path(prior["path"])
    observed = json.loads(path.read_text(encoding="utf-8"))
    assert subject.file_sha256(path) == prior["file_sha256"]
    assert observed["content_sha256"] == prior["content_sha256"]
    assert prior["audit_status"].startswith("BLOCKED_")
    assert prior["preservation_rule"].startswith("V2 is immutable")


def test_dimension_ledger_closes_every_reported_equation() -> None:
    audit = subject.dimension_audit(subject.load_config())
    assert audit["all_pass"] is True
    assert len(audit["rows"]) >= 8
    assert all(row["observed"] == row["expected"] for row in audit["rows"])


def test_dimension_mutation_fails_closed() -> None:
    config = subject.load_config()
    config["unit_system"]["dimensions"]["G"] = [2, -1, -2]
    with pytest.raises(subject.SameLawV3Error, match="dimension audit failed"):
        subject.validate_config(config)


def test_reduced_equations_and_distributional_source_mapping_close() -> None:
    audit = subject.field_equation_and_source_audit(subject.load_config())
    for row in audit["exterior_rows"]:
        assert abs(row["relative_massless_residual"]) < 1e-14
        assert abs(row["relative_massive_residual"]) < 1e-14
    source = audit["distributional_source_mapping"]
    assert source["massless_flux_over_GM"] == pytest.approx(1.0, abs=1e-14)
    assert source["massive_total_over_GM"] == pytest.approx(1.0, abs=1e-14)
    assert audit["full_fierz_pauli_tensor_constraints_claimed"] is False


def test_claim_is_narrower_than_a_full_fierz_pauli_completion() -> None:
    model = subject.load_config()["linear_model"]
    assert "reduced" in model["scope"].lower()
    assert any("full Fierz-Pauli tensor constraints" in item for item in model["not_established"])
    assert any("nonlinear" in item for item in model["not_established"])


def test_one_frozen_state_constructs_metric_and_derivatives() -> None:
    config = subject.load_config()
    state = subject.field_state(0.02, config)
    p = config["physical_parameter_set"]
    model = config["linear_model"]
    assert state.Phi == pytest.approx(
        state.U + model["coefficient_phi_extra"] * p["universal_coupling_g"] * state.Y
    )
    assert state.Psi == pytest.approx(
        state.U + model["coefficient_psi_extra"] * p["universal_coupling_g"] * state.Y
    )
    assert state.Phi_prime == pytest.approx(
        state.U_prime + model["coefficient_phi_extra"] * p["universal_coupling_g"] * state.Y_prime
    )
    assert state.g_tt < 0.0
    assert state.g_space > 0.0


def test_timelike_and_null_observables_consume_the_shared_state(monkeypatch) -> None:
    config = subject.load_config()
    impact = config["fixture_geometry"]["lensing_impacts"][2]
    radius = config["fixture_geometry"]["dynamics_radii"][1]
    baseline_deflection = subject.deflection(impact, config)
    baseline_timelike = subject.timelike_acceleration(radius, config)
    original = subject.field_state

    def changed_psi_state(r, selected_config):
        state = original(r, selected_config)
        return replace(state, Psi_prime=1.1 * state.Psi_prime)

    monkeypatch.setattr(subject, "field_state", changed_psi_state)
    assert subject.deflection(impact, config) != pytest.approx(baseline_deflection, rel=1e-3)
    assert subject.timelike_acceleration(radius, config) == baseline_timelike


def test_no_photon_or_lens_coupling_override_exists_in_public_observable_api() -> None:
    for function in (
        subject.field_state,
        subject.timelike_acceleration,
        subject.radial_null_coordinate_characteristic,
        subject.deflection,
        subject.shapiro_delay,
        subject.endpoint_frequency_ratio,
        subject.image_delay_fixture,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert parameters <= {"r", "impact", "config"}
        assert "coupling" not in parameters
        assert "frequency" not in parameters


def test_linear_timelike_null_delay_and_redshift_formulas_are_consistent() -> None:
    config = subject.load_config()
    radius = config["fixture_geometry"]["dynamics_radii"][1]
    impact = config["fixture_geometry"]["lensing_impacts"][2]
    state = subject.field_state(radius, config)
    c = config["physical_parameter_set"]["c"]
    assert subject.timelike_acceleration(radius, config) == pytest.approx(-state.Phi_prime)
    assert subject.radial_null_coordinate_characteristic(radius, config) == pytest.approx(
        c * (1.0 + (state.Phi + state.Psi) / c**2)
    )
    assert subject.deflection(impact, config) == pytest.approx(0.025996189530254385)
    assert subject.shapiro_delay(impact, config) == pytest.approx(0.0029955634298298845)
    assert 0.0 < subject.endpoint_frequency_ratio(config) < 1.0


def test_fermat_delay_has_explicit_redshift_factor_and_no_separate_knob() -> None:
    config = subject.load_config()
    result = subject.image_delay_fixture(config)
    assert result["negative_image"] < 0.0 < result["positive_image"]
    assert result["lens_redshift_factor"] == 1.0 + config["fixture_geometry"]["lens_redshift"]
    assert result["signed_delay"] == pytest.approx(-0.00019376325648211594)
    assert "separate" not in json.dumps(result).lower()
    assert result["scope"].startswith("local_linear_thin_lens")


def test_tensor_branches_use_one_frozen_inverse_length_but_do_not_overclaim() -> None:
    config = subject.load_config()
    result = subject.tensor_dispersion(config)
    assert result["massless_GR_branch"]["group_speed_over_c"] == 1.0
    massive = result["massive_TT_comparator_branch"]
    assert (
        massive["same_mediator_inverse_length_mu"]
        == config["physical_parameter_set"]["mediator_inverse_length_mu"]
    )
    assert massive["characteristic_speed_over_c"] == 1.0
    assert 0.0 < massive["group_speed_over_c"] < 1.0
    assert "complete constrained" in result["scope"]


def test_two_route_slip_is_nonduplicated_and_honestly_scoped() -> None:
    result = subject.two_route_slip_internal_check(subject.load_config())
    assert len(result["dynamics_rows"]) == 4
    assert len(result["lensing_rows"]) == 4
    assert result["dynamics_generator"].startswith("finite_difference")
    assert result["lensing_generator"].startswith("adaptive_Gauss_Kronrod")
    assert result["lensing_inference_basis"].startswith("independent_composite")
    assert result["fitted_phi_extra_amplitude"] == pytest.approx(0.4, rel=2e-5)
    assert result["fitted_phi_plus_psi_extra_amplitude"] == pytest.approx(0.6, rel=2e-9)
    assert result["reconstructed_gamma_extra"] == pytest.approx(0.5, abs=2e-5)
    assert "not_independent_data_evidence" in result["scope"]


def test_adaptive_rays_match_independent_quadrature_at_every_impact() -> None:
    config = subject.load_config()
    controls = config["numerical_controls"]
    result = subject.numerical_convergence(config)
    assert len(result["all_frozen_impacts"]) == len(config["fixture_geometry"]["lensing_impacts"])
    for row in result["all_frozen_impacts"]:
        assert (
            row["relative_deflection_disagreement"]
            < controls["maximum_relative_deflection_disagreement_each_impact"]
        )
        assert (
            row["relative_shapiro_disagreement"]
            < controls["maximum_relative_shapiro_disagreement_each_impact"]
        )
    assert (
        result["image_delay"]["relative_disagreement"]
        < controls["maximum_relative_image_delay_disagreement"]
    )
    assert result["slip_absolute_error"] < controls["maximum_absolute_slip_error"]


def test_optical_scope_relabels_parity_and_drops_identity_claims() -> None:
    result = subject.optical_scope(subject.load_config())
    assert result["distance_duality_eta"] is None
    assert result["photon_number_survival_fraction"] is None
    assert result["source_observer_Jacobi_reciprocity"] == "NOT_EVALUATED"
    assert len(result["spherical_parity_check_not_reciprocity"]) == 4
    assert all(
        row["deflection_odd_parity_error"] == pytest.approx(0.0, abs=1e-14)
        and row["shapiro_even_parity_error"] == pytest.approx(0.0, abs=1e-14)
        for row in result["spherical_parity_check_not_reciprocity"]
    )


def test_exact_eso_metadata_remains_bound_without_local_payloads() -> None:
    config = subject.load_config()
    manifests = subject._source_manifests(config)
    eso = manifests["ESO325_G004"]
    assert eso["status"].startswith("SOURCE_BLOCKED_")
    assert [row["product_filename"] for row in eso["hst_products"]] == [
        "hst_10429_09_acs_wfc_f814w_j95t09_drc.fits",
        "hst_10429_10_acs_wfc_f475w_j95t10_drc.fits",
    ]
    assert [row["bytes"] for row in eso["hst_products"]] == [369486720, 367663680]
    assert eso["muse_products"][0]["dp_id"] == "ADP.2016-09-07T12:23:32.515"
    assert eso["muse_products"][0]["bytes"] == 7378352640
    assert eso["muse_products"][1]["bytes"] == 1108800
    for row in eso["hst_products"]:
        assert not Path(row["product_filename"]).exists()
    for row in eso["muse_products"]:
        assert not Path(row["original_filename"]).exists()


def test_slacs_split_is_unchanged_and_confirmation_is_unopened() -> None:
    config = subject.load_config()
    manifests = subject._source_manifests(config)
    slacs = manifests["SLACS_HOLDOUT"]
    sample = json.loads(Path(slacs["sample_manifest_path"]).read_text(encoding="utf-8"))
    roles = [row["role"] for row in sample["objects"]]
    assert len(roles) == 57
    assert roles.count("exploration") == 45
    assert roles.count("reserved_confirmation") == 12
    assert slacs["confirmation_opened"] is False
    response_path = next(
        Path(row["path"])
        for row in config["bindings"]
        if row["role"] == "SLACS_UNCHANGED_RESPONSE_MANIFEST"
    )
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["confirmation_objects_requested"] == 0
    assert response["confirmation_response_values_read"] == 0


def test_adversarial_scope_parameter_and_source_mutations_fail_closed() -> None:
    config = subject.load_config()
    mutations = (
        lambda value: value["physical_parameter_set"].__setitem__("photon_multiplier", 1.0),
        lambda value: value["same_law_gate"].__setitem__("all_physical_parameters_shared", False),
        lambda value: value["linear_model"].__setitem__("coefficient_psi_extra", 1.0),
        lambda value: value["linear_model"].__setitem__("not_established", []),
        lambda value: value["source_contract"].__setitem__("ESO325_G004", "READY"),
        lambda value: value["source_contract"].__setitem__("scientific_response_rows_opened", 1),
    )
    for mutation in mutations:
        forged = copy.deepcopy(config)
        mutation(forged)
        with pytest.raises(subject.SameLawV3Error):
            subject.validate_config(forged)


def test_receipt_is_narrow_source_blocked_and_names_next_blocker() -> None:
    receipt = subject.build_receipt()
    assert receipt["status"] == "PASS_NARROW_LINEAR_FIXTURE_BLOCK_REAL_SOURCES"
    assert receipt["internal_same_state_pass"] is True
    assert receipt["photon_only_physical_parameters"] == 0
    assert receipt["source_status"]["scientific_response_rows_opened"] == 0
    assert receipt["next_real_data_blocker"] == "SHA256 for each exact HST and MUSE payload"
    assert "KEEP_ESO_AND_SLACS_RESPONSES_SEALED" in receipt["decision"]


def test_artifacts_are_deterministic_and_explicitly_narrow() -> None:
    config = subject.load_config()
    first = subject.build_artifacts(config)
    second = subject.build_artifacts(config)
    assert first == second
    assert set(first) == {
        "blocked-v2-audit.json",
        "dimension-audit.json",
        "field-equation-and-source-audit.json",
        "derived-linear-observables.json",
        "two-route-slip-internal-check.json",
        "numerical-convergence.json",
        "optical-scope.json",
        "source-and-likelihood-manifests.json",
        "report.md",
    }
    assert b"PASS only for a narrow" in first["report.md"]
    assert b"not evaluated" in first["report.md"]


def test_sealed_packet_matches_deterministic_rebuild() -> None:
    subject.validate_receipt()


def test_atomic_no_clobber(tmp_path) -> None:
    path = tmp_path / "sealed.bin"
    assert subject._atomic_no_clobber(path, b"one") == "CREATED"
    assert subject._atomic_no_clobber(path, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(subject.SameLawV3Error, match="existing artifact differs"):
        subject._atomic_no_clobber(path, b"two")
