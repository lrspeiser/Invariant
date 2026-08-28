from __future__ import annotations

import copy
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.gravity_item2_manga_nonlocal_shape as manga
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
