from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.gravity_item1_effective_dimension as item1
import sigma_theory_compiler.gravity_item2_shape_anisotropy as item2
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / item2.OUTPUT_PATH


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


@pytest.fixture(scope="module")
def prepared() -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    config = dict(item2.load_config(ROOT))
    objects, labels, crosscheck = item2.prepare_shape_objects(ROOT, config)
    return config, objects, labels, crosscheck


def test_real_source_parsers_cover_the_frozen_catalogs() -> None:
    config = item2.load_config(ROOT)
    sources = config["sources"]
    sparc = item2.parse_sparc_properties(
        ROOT / sources["sparc_global_properties"]["files"][0]["path"]
    )
    donahue = item2.parse_donahue_morphology(
        ROOT / sources["clash_xray_morphology"]["file"]["path"]
    )
    zitrin = item2.parse_zitrin_crosscheck(
        ROOT / sources["clash_morphology_crosscheck"]["files"][0]["path"]
    )
    assert len(sparc) == 175
    assert len(donahue) == 25
    assert len(zitrin) == 15
    assert set(item2.DONAHUE_TO_TARGET) <= set(donahue)
    assert len(set(item2.DONAHUE_TO_TARGET.values())) == 20


def test_projected_light_shape_recovers_uniform_disk_aperture_fraction() -> None:
    radius = np.linspace(1.0, 100.0, 100)
    disk = np.ones_like(radius)
    bulge = np.zeros_like(radius)
    shape = item2.projected_light_shape(radius, disk, bulge)
    assert shape["concentration_c20"] == pytest.approx(0.2**2)
    assert shape["bulge_light_fraction"] == 0.0


def test_joined_objects_are_target_blind_and_cover_both_populations(
    prepared: tuple[
        dict[str, object],
        list[dict[str, object]],
        dict[str, dict[str, object]],
        dict[str, object],
    ],
) -> None:
    config, objects, labels, crosscheck = prepared
    expected_features = {
        "axis_ratio_times_concentration",
        "cluster_log_centroid_shift",
        "cluster_log_p30",
        "galaxy_bulge_light_fraction",
        "galaxy_effective_to_disk_radius",
        "projected_axis_ratio",
        "projected_axis_ratio_squared",
        "projected_concentration_c20",
        "support_dimension",
    }
    assert len(objects) == len(labels) == 159
    assert sum(row["domain"] == "galaxy" for row in objects) == 139
    assert sum(row["domain"] == "cluster" for row in objects) == 20
    assert sum(row["point_count"] for row in objects if row["domain"] == "galaxy") == 2720
    assert sum(row["point_count"] for row in objects if row["domain"] == "cluster") == 84
    assert crosscheck["matched_clusters"] == 15
    assert float(crosscheck["axis_ratio_pearson_correlation"]) > 0.9
    assert config["authorization"]["sparc_confirmation_evaluator_accesses_allowed"] == 0
    for row in objects:
        assert set(row["features"]) == expected_features
        assert "observed" not in row["features"]
        assert "sigma" not in row["features"]
        assert 0.0 <= row["features"]["projected_axis_ratio"] <= 1.0
        assert 0.0 < row["features"]["projected_concentration_c20"] < 1.0


def test_folds_hold_out_whole_objects_and_balance_populations(
    prepared: tuple[
        dict[str, object],
        list[dict[str, object]],
        dict[str, dict[str, object]],
        dict[str, object],
    ],
) -> None:
    config, objects, _, _ = prepared
    cv = config["cross_validation"]
    assignments = item1._fold_assignments(objects, salt=cv["fold_salt"], folds=cv["outer_folds"])
    assert set(assignments) == {row["key"] for row in objects}
    for fold in range(5):
        heldout = [row for row in objects if assignments[row["key"]] == fold]
        assert sum(row["domain"] == "cluster" for row in heldout) == 4
        assert sum(row["domain"] == "galaxy" for row in heldout) in {27, 28}


def test_receipt_rebuilds_exactly_and_records_measured_item2_result() -> None:
    stored = _load(OUTPUT)
    rebuilt = item2.build_receipt(ROOT)
    assert rebuilt == stored
    item2.validate_receipt(stored, root=ROOT)
    assert stored["decision"] == "INCONCLUSIVE_ITEM2_SHAPE_ANISOTROPY"
    assert stored["claims"]["roadmap_item_2_complete"] is False
    assert stored["claims"]["universal_projected_shape_predicts_cross_scale_response"] is False
    assert stored["claims"]["alternative_to_gr_established"] is False
    assert stored["claims"]["intrinsic_shape_cause_established"] is False
    assert stored["counts"]["sparc_confirmation_evaluator_accesses"] == 0
    assert stored["counts"]["direct_lensing_likelihood_evaluations"] == 0
    assert stored["counts"]["paid_model_calls"] == 0


def test_nested_universal_selector_cannot_use_population_proxy_models() -> None:
    receipt = _load(OUTPUT)
    forbidden = {
        "linear_support_dimension_proxy",
        "support_plus_shared_shape",
        "domain_specific_shape_bank",
    }
    selected = {
        fold["selected_model_id"] for fold in receipt["nested_universal_shape"]["fold_ledger"]
    }
    assert len(receipt["per_object_diagnostics"]) == 159
    assert len(receipt["nested_universal_shape"]["fold_ledger"]) == 5
    assert selected.isdisjoint(forbidden)
    assert receipt["gate_checks"]["whole_object_target_blind_outer_predictions_complete"] is True
    assert (
        receipt["gate_checks"]["population_proxy_models_excluded_from_universal_admission"] is True
    )
    assert receipt["gate_checks"]["intermediate_or_filamentary_geometry_included"] is False
    assert (
        receipt["gate_checks"]["intrinsic_shape_or_anisotropy_measured_in_both_populations"]
        is False
    )


def test_axis_ratio_overlap_is_measured_in_both_populations() -> None:
    receipt = _load(OUTPUT)
    overlap = receipt["overlap_diagnostic"]
    assert overlap["observed_overlap"] is True
    assert overlap["axis_ratio_interval"][0] <= overlap["axis_ratio_interval"][1]
    assert overlap["by_population"]["galaxy"]["objects"] >= 5
    assert overlap["by_population"]["cluster"]["objects"] >= 5


@pytest.mark.parametrize(
    "claim",
    [
        "alternative_to_gr_established",
        "direct_lensing_test_completed",
        "galaxy_intrinsic_axis_ratio_measured",
        "historical_novelty_established",
        "intrinsic_shape_cause_established",
        "sequential_G6_G7_G8_advanced",
        "sparc_confirmation_opened",
    ],
)
def test_resealed_overclaim_is_rejected(claim: str) -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["claims"][claim] = True
    with pytest.raises(item2.GravityItem2ShapeAnisotropyError):
        item2.validate_receipt(_reseal(receipt), root=ROOT)


def test_resealed_proxy_admission_and_confirmation_access_are_rejected() -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["model_results"]["linear_support_dimension_proxy"][
        "qualifying_universal_shape_model"
    ] = True
    with pytest.raises(item2.GravityItem2ShapeAnisotropyError):
        item2.validate_receipt(_reseal(receipt), root=ROOT)
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["counts"]["sparc_confirmation_evaluator_accesses"] = 1
    with pytest.raises(item2.GravityItem2ShapeAnisotropyError):
        item2.validate_receipt(_reseal(receipt), root=ROOT)


def test_resealed_decision_must_agree_with_measured_gates() -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    key = "intermediate_or_filamentary_geometry_included"
    receipt["gate_checks"][key] = not receipt["gate_checks"][key]
    with pytest.raises(item2.GravityItem2ShapeAnisotropyError):
        item2.validate_receipt(_reseal(receipt), root=ROOT)
