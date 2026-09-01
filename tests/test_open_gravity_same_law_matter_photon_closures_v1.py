from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_same_law_matter_photon_closures_v1 as subject


def _raw_config() -> dict[str, object]:
    return json.loads(subject.CONFIG_PATH.read_text(encoding="utf-8"))


def _closure(config: dict[str, object], closure_id: str) -> dict[str, object]:
    return next(row for row in config["closures"] if row["id"] == closure_id)  # type: ignore[index,union-attr]


def _fixture(fixture_id: str, case: str) -> dict[str, object]:
    return next(
        row
        for row in subject.synthetic_fixtures()
        if row["fixture_id"] == fixture_id and row["case"] == case
    )


def test_package_hash_pins_are_exact() -> None:
    assert subject.file_sha256(subject.CONFIG_PATH) == subject._CONFIG_RAW_SHA256
    assert subject.content_sha256(_raw_config()) == subject._CONFIG_CONTENT_SHA256
    assert subject.module_semantic_sha256() == subject._MODULE_SEMANTIC_SHA256
    assert subject.file_sha256(subject.TEST_PATH) == subject._TEST_RAW_SHA256


def test_taxonomy_channels_sources_controls_and_response_gate() -> None:
    config = subject.load_config()
    assert tuple(row["id"] for row in config["closures"]) == subject._CLOSURE_IDS
    assert tuple(config["channel_names"]) == subject._CHANNELS
    assert len(config["closures"]) == 16
    assert len({row["family"] for row in config["closures"]}) >= 14
    assert len(config["published_sources"]) == 10
    assert len(config["mandatory_controls"]) == 12
    assert all(
        config["real_data_preflight"][stage]["response_status"] == "NOT_DOWNLOADED_NOT_SCORED"
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "OPEN"),
        (("claim_boundary", "real_data_scored"), True),
        (("access_contract", "paid_calls"), 1),
        (("real_data_preflight", "stage_2", "response_status"), "SCORED"),
        (("outputs", "receipt"), "elsewhere.json"),
    ],
)
def test_semantic_mutations_reject(path: tuple[str, ...], replacement: object) -> None:
    config = copy.deepcopy(_raw_config())
    target = config
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(subject.SameLawClosureError):
        subject.validate_config(config)


def test_slip_reconstruction_is_exact_for_every_single_metric_fixture() -> None:
    config = subject.load_config()
    for closure in config["closures"]:
        if not closure["photon_mode"].startswith("SINGLE_"):
            continue
        if subject._coverage(closure)["photon_characteristic"] == "BLOCKED":
            continue
        for fixture in subject.synthetic_fixtures():
            row = subject.predict(closure, fixture)
            reconstructed = subject.reconstruct_spatial_potential_gradient(
                row["matter_acceleration"], row["lensing_acceleration"]
            )
            assert reconstructed == pytest.approx(row["psi_gradient"], abs=1.0e-14)
            assert row["deflection"] == pytest.approx(
                2.0 * row["lensing_acceleration"], abs=1.0e-14
            )


def test_conformal_scalar_changes_matter_but_cancels_direct_lensing() -> None:
    config = subject.load_config()
    fixture = _fixture("F02_EXTRA_DOMINATED", "extra")
    gr = subject.predict(_closure(config, "L00_GR_BARYON_METRIC"), fixture)
    conformal = subject.predict(_closure(config, "L02_UNIVERSAL_CONFORMAL_SCALAR"), fixture)
    assert conformal["matter_acceleration"] != gr["matter_acceleration"]
    assert conformal["lensing_acceleration"] == pytest.approx(gr["lensing_acceleration"])
    assert conformal["shapiro_delay"] == pytest.approx(gr["shapiro_delay"])


def test_disformal_and_nonlocal_metric_tie_lensing_to_matter() -> None:
    config = subject.load_config()
    fixture = _fixture("F02_EXTRA_DOMINATED", "extra")
    for closure_id in (
        "L04_TEVES_DISFORMAL_VECTOR_SCALAR",
        "L06_SPATIAL_NONLOCAL_SINGLE_METRIC",
        "L08_CAUSAL_METRIC_MEMORY",
    ):
        row = subject.predict(_closure(config, closure_id), fixture)
        assert row["lensing_acceleration"] == pytest.approx(row["matter_acceleration"])


def test_tied_constitutive_chromatic_residual_has_frozen_ratio() -> None:
    config = subject.load_config()
    closure = _closure(config, "L05_TIED_REFRACTIVE_CONSTITUTIVE")
    low = subject.predict(closure, _fixture("F06_TWO_FREQUENCIES", "low"))
    high = subject.predict(closure, _fixture("F06_TWO_FREQUENCIES", "high"))
    metric_low = 0.5 * (low["matter_acceleration"] + low["psi_gradient"])
    metric_high = 0.5 * (high["matter_acceleration"] + high["psi_gradient"])
    assert (low["lensing_acceleration"] - metric_low) / (
        high["lensing_acceleration"] - metric_high
    ) == pytest.approx(16.0)
    assert low["chromatic_log_slope"] == -2.0


def test_path_memory_has_equal_endpoint_path_and_duality_falsifier() -> None:
    config = subject.load_config()
    closure = _closure(config, "L07_TIED_PATH_MEMORY_CHARACTERISTIC")
    short = subject.predict(closure, _fixture("F04_EQUAL_ENDPOINT_TWO_PATHS", "short"))
    long = subject.predict(closure, _fixture("F04_EQUAL_ENDPOINT_TWO_PATHS", "long"))
    assert short["matter_acceleration"] == long["matter_acceleration"]
    assert short["gravitational_redshift"] == long["gravitational_redshift"]
    assert short["shapiro_delay"] != long["shapiro_delay"]
    assert short["distance_duality_eta"] != long["distance_duality_eta"]


def test_causal_metric_memory_phase_at_same_instantaneous_dynamics() -> None:
    config = subject.load_config()
    closure = _closure(config, "L08_CAUSAL_METRIC_MEMORY")
    rising = subject.predict(closure, _fixture("F05_SAME_STATE_MEMORY_PHASE", "rising"))
    falling = subject.predict(closure, _fixture("F05_SAME_STATE_MEMORY_PHASE", "falling"))
    assert rising["matter_acceleration"] == falling["matter_acceleration"]
    assert rising["lensing_acceleration"] == falling["lensing_acceleration"]
    assert rising["gravitational_redshift"] > falling["gravitational_redshift"]


def test_massive_mediators_have_range_and_spin2_group_dispersion() -> None:
    config = subject.load_config()
    scalar = _closure(config, "L09_MASSIVE_CONFORMAL_SCALAR")
    near = subject.predict(scalar, _fixture("F07_MEDIATOR_RANGE", "near"))
    far = subject.predict(scalar, _fixture("F07_MEDIATOR_RANGE", "far"))
    assert near["matter_acceleration"] > far["matter_acceleration"]
    spin2 = _closure(config, "L11_MASSIVE_SPIN2_FIXED_SLIP")
    low_k = subject.predict(spin2, _fixture("F08_TENSOR_DISPERSION", "low_k"))
    high_k = subject.predict(spin2, _fixture("F08_TENSOR_DISPERSION", "high_k"))
    assert low_k["tensor_characteristic_speed_over_c"] == 1.0
    assert low_k["tensor_group_speed_over_c"] < high_k["tensor_group_speed_over_c"] < 1.0


def test_incomplete_and_forbidden_controls_are_retained_not_silently_completed() -> None:
    config = subject.load_config()
    pure_photon = _closure(config, "L12_PURE_PHOTON_REFRACTION_FAILURE")
    acceleration_only = _closure(config, "L13_MASSIVE_ONLY_ACCELERATION_FAILURE")
    aqual = _closure(config, "L15_NONRELATIVISTIC_AQUAL_ONLY")
    assert pure_photon["same_constants"] is False
    assert subject._coverage(pure_photon)["massive_particle_acceleration"].endswith("FAIL")
    for closure in (acceleration_only, aqual):
        coverage = subject._coverage(closure)
        assert coverage["massive_particle_acceleration"] == "DERIVED"
        assert all(coverage[name] == "BLOCKED" for name in subject._CHANNELS[1:])


def test_exact_gradient_rewrite_remains_equivalent_to_gr() -> None:
    classes = subject.equivalence_classes(subject.load_config())
    group = next(group for group in classes if "L00_GR_BARYON_METRIC" in group)
    assert "L14_EXACT_GRADIENT_PATH_REWRITE" in group


def test_receipt_and_artifacts_are_deterministic_and_claim_bounded() -> None:
    first = subject.build_receipt()
    second = subject.build_receipt()
    assert first == second
    assert subject.build_artifacts(subject.load_config()) == subject.build_artifacts(
        subject.load_config()
    )
    assert first["summary"]["closures"] == 16
    assert first["summary"]["synthetic_fixtures"] == 13
    assert first["summary"]["synthetic_prediction_rows"] == 208
    assert first["summary"]["real_response_rows_scored"] == 0
    assert first["claim_boundary"]["historical_novelty_established"] is False
    report = subject.build_artifacts(subject.load_config())["report.md"].decode()
    assert "Psi' = 2 g_lens - g_dyn" in report
    assert "gamma=0.97 +/- 0.09" in report
    assert "no payload was downloaded" in report


def test_written_packet_checks_and_is_append_only() -> None:
    subject.validate_receipt()
    assert subject.write_packet() == "EXISTING_IDENTICAL"
    assert Path(subject.OUTPUT_PATH).is_file()


def test_access_and_response_boundaries_are_zero() -> None:
    config = subject.load_config()
    access = config["access_contract"]
    for key in (
        "raw_scientific_payloads_downloaded",
        "scientific_response_rows_opened",
        "scientific_response_rows_scored",
        "network_calls_by_builder",
        "external_model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        assert access[key] == 0
    assert config["claim_boundary"]["real_data_scored"] is False
