from __future__ import annotations

import copy
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.gravity_item2_manga_nonlocal_shape as manga
import sigma_theory_compiler.gravity_item2_manga_nonlocal_shape_experiment as experiment
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / manga.SAMPLE_MANIFEST_PATH


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def test_attempt_four_is_frozen_before_selected_kinematics() -> None:
    config = manga.load_config(ROOT)
    assert config["status"] == "frozen_before_selected_kinematic_map_access"
    assert config["authorization"]["selected_exploration_kinematic_maps_allowed"] is True
    assert config["authorization"]["reserved_confirmation_kinematic_maps_allowed"] is False
    assert config["target_blind_sample"]["reserved_confirmation_target_accesses_allowed"] == 0
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert "JAM or DynPop total mass" in config["aperture_and_response"]["forbidden_targets"]
    assert "lensing-derived mass" in config["aperture_and_response"]["forbidden_targets"]


def test_sample_is_catalog_deterministic_balanced_and_target_blind() -> None:
    config = manga.load_config(ROOT)
    manifest = _load(SAMPLE)
    manga.validate_sample_manifest(manifest, config=config)
    assert manifest["decision"] == "PASS_TARGET_BLIND_SAMPLE_SELECTION"
    assert manifest["selection_boundary"]["selected_pca_files_opened"] == 0
    assert manifest["selection_boundary"]["selected_dap_maps_opened"] == 0
    assert manifest["selection_boundary"]["selected_kinematic_values_read"] == 0
    assert manifest["selection_boundary"]["reserved_confirmation_target_accesses"] == 0
    assert manifest["counts"]["source_endpoint_queries"] == 0
    objects = manifest["objects"]
    assert len(objects) == 90
    assert len({row["plateifu"] for row in objects}) == 90
    assert len({row["manga_id"] for row in objects}) == 90
    roles = Counter(row["role"] for row in objects)
    assert roles == {"exploration": 60, "reserved_confirmation": 30}
    strata = Counter((row["visual_class"], row["axis_bin"], row["role"]) for row in objects)
    for visual_class in (1, 2):
        for axis_bin in range(3):
            assert strata[(visual_class, axis_bin, "exploration")] == 10
            assert strata[(visual_class, axis_bin, "reserved_confirmation")] == 5


def test_shape_extractor_cannot_accept_a_dap_target_path() -> None:
    signature = inspect.signature(manga.measure_shape_only)
    assert tuple(signature.parameters) == ("pca_path", "object_row", "config")
    source = inspect.getsource(manga.measure_shape_only)
    assert "STELLAR_VEL" not in source
    assert "STELLAR_SIGMA" not in source
    assert "dap" not in source.lower()


def test_projected_moments_are_rotation_and_reflection_invariant() -> None:
    y, x = np.indices((81, 81), dtype=np.float64)
    dx = x - 40.0
    dy = y - 40.0
    mass = np.exp(-0.5 * ((dx / 10.0) ** 2 + (dy / 5.0) ** 2))
    mass *= 1.0 + 0.08 * np.cos(3.0 * np.arctan2(dy, dx))
    valid = np.ones_like(mass, dtype=bool)

    def measured(image: np.ndarray) -> tuple[float, float, float]:
        geometry = manga._ellipse_geometry(image, valid, 24.0, 0.5)
        moments = manga._aperture_moments(image, valid, geometry, 24.0)
        return moments["quadrupole"], moments["m3"], moments["m4"]

    primary = measured(mass)
    rotated = measured(np.rot90(mass))
    reflected = measured(np.fliplr(mass))
    assert rotated == pytest.approx(primary, rel=0, abs=1.0e-12)
    assert reflected == pytest.approx(primary, rel=0, abs=1.0e-12)


@pytest.mark.parametrize(
    "claim",
    [
        "confirmation_opened",
        "kinematic_response_seen_during_selection",
        "roadmap_item_2_complete",
        "alternative_to_gr_established",
    ],
)
def test_resealed_sample_overclaim_is_rejected(claim: str) -> None:
    config = manga.load_config(ROOT)
    manifest = copy.deepcopy(_load(SAMPLE))
    manifest["claims"][claim] = True
    with pytest.raises(manga.GravityItem2MangaNonlocalShapeError):
        manga.validate_sample_manifest(_reseal(manifest), config=config)


def test_acquisition_and_extraction_never_touch_confirmation() -> None:
    config = manga.load_config(ROOT)
    sample = _load(SAMPLE)
    source_manifest = _load(ROOT / manga.SOURCE_MANIFEST_PATH)
    manga.validate_source_manifest(source_manifest, config=config, sample=sample)
    assert source_manifest["decision"] == "PASS_EXPLORATION_SOURCE_ACQUISITION"
    assert source_manifest["boundary"]["exploration_objects"] == 60
    assert source_manifest["boundary"]["reserved_confirmation_objects_acquired"] == 0
    assert source_manifest["boundary"]["reserved_confirmation_target_accesses"] == 0
    assert {row["plateifu"] for row in source_manifest["records"]} == {
        row["plateifu"] for row in sample["objects"] if row["role"] == "exploration"
    }
    extraction = _load(ROOT / manga.EXTRACTION_SUMMARY_PATH)
    assert extraction["decision"] == "FAIL_EXPLORATION_QUALITY"
    assert extraction["counts"] == {
        "quality_failures": 5,
        "quality_passing": 55,
        "reserved_confirmation_target_accesses": 0,
        "selected_exploration": 60,
    }


def test_feature_table_preserves_both_classes_and_all_shape_strata() -> None:
    config = manga.load_config(ROOT)
    rows = experiment._load_feature_rows(ROOT, config)
    assert len(rows) == 55
    assert Counter(row["visual_class"] for row in rows) == {1: 29, 2: 26}
    assert {(row["visual_class"], row["axis_bin"]) for row in rows} == {
        (visual_class, axis_bin) for visual_class in (1, 2) for axis_bin in range(3)
    }
    assert all(row["unique_kinematic_bins"] >= 12 for row in rows)
    assert all(row["usable_kinematic_luminosity_fraction"] >= 0.6 for row in rows)


def test_outer_folds_hold_out_whole_galaxies_and_retain_every_stratum() -> None:
    config = manga.load_config(ROOT)
    rows = experiment._load_feature_rows(ROOT, config)
    assignments = experiment.fold_assignments(
        rows,
        salt=config["cross_validation"]["fold_salt"],
        folds=config["cross_validation"]["outer_folds"],
    )
    assert set(assignments) == {row["plateifu"] for row in rows}
    for fold in range(5):
        heldout = [row for row in rows if assignments[row["plateifu"]] == fold]
        assert len(heldout) >= 9
        assert {row["visual_class"] for row in heldout} == {1, 2}
        assert {(row["visual_class"], row["axis_bin"]) for row in heldout} == {
            (visual_class, axis_bin) for visual_class in (1, 2) for axis_bin in range(3)
        }


def test_receipt_replays_and_records_negative_shape_result() -> None:
    stored = _load(ROOT / experiment.OUTPUT_PATH)
    assert experiment.build_receipt(ROOT) == stored
    experiment.validate_receipt(stored, root=ROOT)
    assert stored["decision"] == "INCONCLUSIVE_ITEM2_MANGA_NONLOCAL_SHAPE_QUALITY_GATE"
    assert stored["counts"]["exploration_quality_passing"] == 55
    assert stored["counts"]["reserved_confirmation_target_accesses"] == 0
    assert float(stored["primary_response"]["selected_metrics"]["overall"]["r2"]) < 0
    assert (
        float(
            stored["primary_response"]["baseline_metrics"]["mass_size_nuisance"][
                "overall"
            ]["r2"]
        )
        > 0
    )
    assert stored["gate_checks"]["reserved_confirmation_untouched"] is True
    assert sum(stored["gate_checks"].values()) == 1


@pytest.mark.parametrize("claim", list(_load(ROOT / experiment.OUTPUT_PATH)["claims"]))
def test_resealed_receipt_overclaim_is_rejected(claim: str) -> None:
    receipt = copy.deepcopy(_load(ROOT / experiment.OUTPUT_PATH))
    receipt["claims"][claim] = True
    with pytest.raises(experiment.GravityItem2MangaNonlocalShapeExperimentError):
        experiment.validate_receipt(_reseal(receipt), root=ROOT)


def test_resealed_false_pass_is_rejected() -> None:
    receipt = copy.deepcopy(_load(ROOT / experiment.OUTPUT_PATH))
    receipt["decision"] = "PASS_ITEM2_MANGA_EXPLORATION_AWAITING_CONFIRMATION_AUTHORIZATION"
    with pytest.raises(experiment.GravityItem2MangaNonlocalShapeExperimentError):
        experiment.validate_receipt(_reseal(receipt), root=ROOT)
