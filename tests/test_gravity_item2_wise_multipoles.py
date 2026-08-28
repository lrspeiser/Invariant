from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

import sigma_theory_compiler.gravity_item1_effective_dimension as item1
import sigma_theory_compiler.gravity_item2_wise_multipole_experiment as experiment
import sigma_theory_compiler.gravity_item2_wise_multipoles as wise
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / experiment.OUTPUT_PATH


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def _synthetic_wcs(size: int, *, pixel_scale_arcsec: float = 1.0) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(size + 1) / 2.0, (size + 1) / 2.0]
    wcs.wcs.cdelt = [-pixel_scale_arcsec / 3600.0, pixel_scale_arcsec / 3600.0]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _synthetic_source(*, axis_ratio: float) -> tuple[np.ndarray, WCS]:
    size = 161
    yy, xx = np.indices((size, size), dtype=np.float64)
    center = (size - 1) / 2.0
    angle = np.deg2rad(27.0)
    dx = xx - center
    dy = yy - center
    major = dx * np.cos(angle) + dy * np.sin(angle)
    minor = -dx * np.sin(angle) + dy * np.cos(angle)
    profile = 80.0 * np.exp(-0.5 * ((major / 11.0) ** 2 + (minor / (11.0 * axis_ratio)) ** 2))
    noise = np.random.default_rng(20260827).normal(0.0, 0.25, profile.shape)
    return profile + noise + 3.0, _synthetic_wcs(size)


def test_target_blind_population_and_source_contract_are_frozen() -> None:
    config = wise.load_extraction_config(ROOT)
    galaxies = wise._eligible_galaxies(ROOT, config)
    assert len(galaxies) == 83
    assert len({row["name"] for row in galaxies}) == 83
    assert all(row["inclination_deg"] <= 65.0 for row in galaxies)
    assert all(
        set(row)
        == {
            "name",
            "ra_deg",
            "dec_deg",
            "distance_mpc",
            "inclination_deg",
            "effective_radius_kpc",
        }
        for row in galaxies
    )
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["authorization"]["sparc_confirmation_evaluator_accesses_allowed"] == 0
    assert config["image_extraction"]["target_fields_available_to_feature_computation"] is False


def test_synthetic_circular_and_barred_sources_have_ordered_quadrupoles() -> None:
    config = wise.load_extraction_config(ROOT)
    circular, wcs = _synthetic_source(axis_ratio=1.0)
    barred, _ = _synthetic_source(axis_ratio=0.42)
    common = {
        "ra_deg": 150.0,
        "dec_deg": 2.0,
        "aperture_arcsec": 40.0,
        "inclination_deg": 0.0,
        "extraction": config["image_extraction"],
    }
    circular_result = wise.measure_w1_multipoles(circular, wcs, **common)
    barred_result = wise.measure_w1_multipoles(barred, wcs, **common)
    assert circular_result["measurement_valid"] is True
    assert circular_result["image_quality_pass"] is True
    assert barred_result["image_quality_pass"] is True
    assert circular_result["quadrupole_amplitude"] < 0.03
    assert barred_result["quadrupole_amplitude"] > 0.25
    assert barred_result["quadrupole_amplitude"] > 8 * circular_result["quadrupole_amplitude"]
    assert circular_result["m3_aperture_amplitude"] < 0.01


def test_measurement_failure_can_be_retained_but_never_passes_quality() -> None:
    failed = wise._failed_measurement("zero central unWISE flux")
    assert failed["measurement_valid"] is False
    assert failed["image_quality_pass"] is False
    assert failed["quality_failure_reason"].startswith("measurement_error:")
    assert set(failed) <= set(wise.FEATURE_COLUMNS)


def test_sealed_real_extraction_covers_every_eligible_image_and_quality_flag() -> None:
    manifest = wise.validate_extraction(ROOT)
    assert manifest["counts"]["eligible_galaxies"] == 83
    assert manifest["counts"]["images"] == 83
    assert manifest["counts"]["measurement_valid"] == 79
    assert manifest["counts"]["quality_pass_galaxies"] == 68
    assert manifest["counts"]["target_fields_used_by_feature_computation"] == 0
    assert manifest["source_bindings"]["extractor"]["sha256"] == wise._sha256(
        ROOT / wise.SOURCE_PATH
    )


def test_extractor_source_has_no_gravity_targets_or_item1_beta_rows() -> None:
    source = (ROOT / wise.SOURCE_PATH).read_text(encoding="utf-8")
    assert "per_object_diagnostics" not in source
    assert "oracle_beta" not in source
    assert "Vflat" not in source
    assert "observed" not in source


def test_joined_real_objects_cover_quality_galaxies_and_all_clusters() -> None:
    config = experiment.load_config(ROOT)
    objects, labels, manifest = experiment.prepare_multipole_objects(ROOT, config)
    expected_features = {
        "centroid_shift",
        "concentration_c20",
        "concentration_times_energy",
        "log1p_multipole_energy_over_0p05",
        "m3_aperture_amplitude",
        "m4_aperture_amplitude",
        "multipole_energy",
        "multipole_energy_squared",
        "quadrupole_amplitude",
        "support_dimension",
    }
    assert len(objects) == len(labels) == 88
    assert sum(row["domain"] == "galaxy" for row in objects) == 68
    assert sum(row["domain"] == "cluster" for row in objects) == 20
    assert sum(row["point_count"] for row in objects if row["domain"] == "galaxy") == 1395
    assert sum(row["point_count"] for row in objects if row["domain"] == "cluster") == 84
    assert manifest["counts"]["quality_pass_galaxies"] == 68
    assert all(set(row["features"]) == expected_features for row in objects)
    assert all("observed" not in row["features"] for row in objects)


def test_cluster_power_ratio_conversion_uses_frozen_common_grammar() -> None:
    morphology = {
        "axis_ratio": 0.8,
        "concentration": 0.2,
        "centroid_shift": 0.01,
        "p30": 1.0e-7,
        "p40": 2.0e-8,
    }
    features = experiment.cluster_multipole_features(morphology)
    assert features["quadrupole_amplitude"] == pytest.approx((1 - 0.8**2) / (1 + 0.8**2))
    assert features["m3_aperture_amplitude"] == pytest.approx(np.sqrt(18.0e-7) * abs(np.log(500.0)))
    assert features["m4_aperture_amplitude"] == pytest.approx(
        np.sqrt(32.0 * 2.0e-8) * abs(np.log(500.0))
    )


def test_folds_hold_out_whole_objects_and_balance_populations() -> None:
    config = experiment.load_config(ROOT)
    objects, _, _ = experiment.prepare_multipole_objects(ROOT, config)
    cv = config["cross_validation"]
    assignments = item1._fold_assignments(objects, salt=cv["fold_salt"], folds=cv["outer_folds"])
    assert set(assignments) == {row["key"] for row in objects}
    for fold in range(5):
        heldout = [row for row in objects if assignments[row["key"]] == fold]
        assert sum(row["domain"] == "cluster" for row in heldout) == 4
        assert sum(row["domain"] == "galaxy" for row in heldout) in {13, 14}


def test_receipt_rebuilds_exactly_and_records_measured_second_attempt() -> None:
    stored = _load(OUTPUT)
    rebuilt = experiment.build_receipt(ROOT)
    assert rebuilt == stored
    experiment.validate_receipt(stored, root=ROOT)
    assert stored["decision"] == "INCONCLUSIVE_ITEM2_WISE_MULTIPOLES"
    assert stored["claims"]["roadmap_item_2_complete"] is False
    assert stored["claims"]["alternative_to_gr_established"] is False
    assert stored["counts"]["paid_model_calls"] == 0
    assert stored["counts"]["sparc_confirmation_evaluator_accesses"] == 0
    assert stored["counts"]["direct_lensing_likelihood_evaluations"] == 0


def test_measured_result_preserves_external_and_predictive_counterexamples() -> None:
    receipt = _load(OUTPUT)
    assert receipt["external_s4g_validation"]["primary_matches"] == 22
    assert float(receipt["external_s4g_validation"]["quadrupole_vs_s4g_family_spearman"]) < 0.0
    assert receipt["gate_checks"]["published_s4g_bar_validation_positive"] is False
    assert receipt["gate_checks"]["universal_beta_r2_positive_in_each_population"] is False
    assert (
        receipt["gate_checks"][
            "universal_model_beats_support_proxy_in_energy_overlap_for_each_population"
        ]
        is False
    )
    assert receipt["counts"]["intermediate_bar_like_galaxies"] == 33


@pytest.mark.parametrize(
    "claim",
    [
        "alternative_to_gr_established",
        "direct_lensing_test_completed",
        "historical_novelty_established",
        "intrinsic_shape_cause_established",
        "sequential_G6_G7_G8_advanced",
        "sparc_confirmation_opened",
        "wise_w1_is_pure_baryonic_mass_map",
    ],
)
def test_resealed_overclaim_is_rejected(claim: str) -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["claims"][claim] = True
    with pytest.raises(experiment.GravityItem2WiseMultipoleExperimentError):
        experiment.validate_receipt(_reseal(receipt), root=ROOT)


def test_resealed_proxy_admission_and_gate_decision_drift_are_rejected() -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["model_results"]["linear_support_dimension_proxy"][
        "qualifying_universal_multipole_model"
    ] = True
    with pytest.raises(experiment.GravityItem2WiseMultipoleExperimentError):
        experiment.validate_receipt(_reseal(receipt), root=ROOT)

    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["gate_checks"]["published_s4g_bar_validation_positive"] = True
    receipt["decision"] = "PASS_ITEM2_WISE_MULTIPOLE_DEVELOPMENT_GATE"
    with pytest.raises(experiment.GravityItem2WiseMultipoleExperimentError):
        experiment.validate_receipt(_reseal(receipt), root=ROOT)


@pytest.mark.parametrize(
    "name, expected",
    [("NGC0024", "NGC24"), ("ngc 024", "NGC24"), ("UGC00128", "UGC128"), ("F563-V2", "F563V2")],
)
def test_name_normalization_is_deterministic(name: str, expected: str) -> None:
    assert wise.normalize_galaxy_name(name) == expected
