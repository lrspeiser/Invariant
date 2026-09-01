from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from sigma_theory_compiler import open_gravity_rg_holmberg_ii_things_2d_fixed_score_v1 as score


def _header() -> fits.Header:
    header = fits.Header()
    header["NAXIS1"] = 4
    header["NAXIS2"] = 4
    header["CDELT1"] = -0.001
    header["CDELT2"] = 0.001
    return header


def test_config_and_sealed_prediction_evidence_are_valid() -> None:
    config = score.load_config(verify_package=False)
    score.validate_config(config)
    prediction_config, receipt, manifest = score._load_prediction_evidence(config)
    assert prediction_config["execution_contract"]["candidate_resolution_predictions"] == 72
    assert receipt["all_solver_gates_pass"] is True
    assert manifest["array_file_count"] == 117


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("prediction_binding", "all_solver_gates_pass", False),
        ("prediction_binding", "response_pixels_used_to_build_predictions", 1),
        ("response_contract", "minimum_dispersion_scale_m_s", 0.0),
        ("score_contract", "primary_cell_id", "AFTER_LOOKING"),
        ("score_contract", "per_model_sign_selection", True),
        ("score_contract", "source_or_geometry_reselection", True),
        ("score_contract", "p_values_computed", True),
        ("claim_boundary", "publication_ready", True),
    ],
)
def test_material_mutations_fail(section: str, key: str, value: object) -> None:
    config = copy.deepcopy(score.load_config(verify_package=False))
    config[section][key] = value
    with pytest.raises(score.HolmbergScoreError):
        score.validate_config(config)


def test_preflight_has_exact_four_response_assets() -> None:
    config = score.load_config(verify_package=False)
    preflight = score._load_preflight(config)
    rows = score._response_rows(preflight)
    assert set(rows) == {
        ("NATURAL", "MOM1"),
        ("NATURAL", "MOM2"),
        ("ROBUST", "MOM1"),
        ("ROBUST", "MOM2"),
    }


def test_primary_metrics_use_shared_nuisance_not_model_specific_offset() -> None:
    observed = np.asarray([100.0, 102.0, 104.0, 106.0])
    dispersion = np.full(4, 10.0)
    predicted = np.asarray([-3.0, -1.0, 1.0, 3.0])
    mask = np.ones(4, dtype=bool)
    metrics = score._model_metrics(
        observed,
        dispersion,
        predicted,
        mask,
        sign=1.0,
        shared_systemic=100.0,
        minimum_dispersion=3.0,
    )
    assert metrics["shared_systemic_velocity_m_s"] == 100.0
    assert metrics["rmse_m_s"] > metrics["model_specific_best_offset_diagnostic"]["rmse_m_s"]
    assert metrics["model_specific_best_offset_diagnostic"]["offset_m_s"] == pytest.approx(103.0)
    assert metrics["residual_count"] == 4


def test_common_mask_is_identical_for_all_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    eligibility = np.asarray([[1, 1], [1, 0]], dtype=np.uint8)
    candidate_arrays = {
        candidate: np.asarray([[1.0, 2.0], [3.0, 4.0]]) for candidate in score._CANDIDATES
    }

    def load(_manifest, _cell_id, role):
        if role == "natural_eligibility":
            return eligibility
        candidate = role.removesuffix("__NATURAL")
        return candidate_arrays[candidate]

    monkeypatch.setattr(score, "_load_prediction_array", load)
    observed = np.asarray([[10.0, np.nan], [12.0, 13.0]])
    dispersion = np.asarray([[5.0, 5.0], [0.0, 5.0]])
    mask, arrays = score._common_mask({}, "CELL", "NATURAL", observed, dispersion)
    assert mask.tolist() == [[True, False], [False, False]]
    assert set(arrays) == set(score._CANDIDATES)


def test_nuisance_is_one_observed_value_per_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.asarray([[90.0, 100.0], [110.0, 120.0]])
    dispersion = np.ones((2, 2))
    header = _header()
    responses = {resolution: (observed, dispersion, header) for resolution in score._RESOLUTIONS}
    mask = np.ones((2, 2), dtype=bool)
    monkeypatch.setattr(
        score,
        "_common_mask",
        lambda *_args: (mask, {candidate: np.zeros((2, 2)) for candidate in score._CANDIDATES}),
    )
    monkeypatch.setattr(
        score.predictions,
        "_world_grid",
        lambda _header: (np.zeros((2, 2)), np.zeros((2, 2))),
    )
    major = np.asarray([[-1.0, -0.5], [0.5, 1.0]])
    monkeypatch.setattr(
        score.predictions,
        "_disk_sky_coordinates",
        lambda *_args: (major, np.zeros_like(major), np.ones_like(major), major),
    )
    nuisance = score._nuisance_values(
        {}, {"cell_run_id": score._REFERENCE_CELL_ID, "geometry": {}}, responses
    )
    assert nuisance["NATURAL"]["shared_systemic_velocity_m_s"] == 105.0
    assert nuisance["ROBUST"]["shared_systemic_velocity_m_s"] == 105.0
    assert nuisance["NATURAL"]["rotation_sign"] == 1.0


def test_cell_score_ranks_all_four_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.asarray([[8.0, 10.0], [12.0, 14.0]])
    dispersion = np.full((2, 2), 4.0)
    mask = np.ones((2, 2), dtype=bool)
    model_arrays = {
        "NEWTON_3D_DST": np.zeros((2, 2)),
        "RAR_2016_ON_NEWTON_3D": np.asarray([[-1.0, 0.0], [1.0, 2.0]]),
        "MOND_STANDARD_MU_ON_NEWTON_3D": np.asarray([[-2.0, 0.0], [2.0, 4.0]]),
        "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG": np.asarray([[-3.0, -1.0], [1.0, 3.0]]),
    }
    monkeypatch.setattr(score, "_common_mask", lambda *_args: (mask, model_arrays))
    preflight = {
        "response_assets": [{"resolution": "NATURAL", "observable": "MOM1", "sha256": "a" * 64}],
        "exact_header_contract": {"natural_beam_deg": [0.01, 0.008, 0.0]},
    }
    config = score.load_config(verify_package=False)
    cell = {
        "cell_run_id": score._REFERENCE_CELL_ID,
        "conversion_cell_id": "IRAC1_FIXED_ML0P6",
        "geometry": {"geometry_variant_id": "I38P0", "inclination_deg": 38.0},
    }
    result = score._score_cell(
        config,
        {},
        cell,
        "NATURAL",
        {"NATURAL": (observed, dispersion, _header())},
        {
            "NATURAL": {
                "rotation_sign": 1.0,
                "shared_systemic_velocity_m_s": 11.0,
                "major_axis_velocity_covariance_kpc_m_s": 1.0,
                "reference_common_pixel_count": 4,
            }
        },
        preflight,
    )
    assert result["winner"] == "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"
    assert result["rg_beats_all_three_comparators"] is True
    assert len(result["rmse_ranking"]) == 4


def test_atomic_output_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    assert score._atomic_no_clobber(path, b"fixed") == "CREATED"
    assert score._atomic_no_clobber(path, b"fixed") == "EXISTING_IDENTICAL"
    with pytest.raises(score.HolmbergScoreError):
        score._atomic_no_clobber(path, b"changed")


def test_output_path_is_fixed_and_any_existing_result_is_canonical() -> None:
    assert score.load_config(verify_package=False)["output_path"] == score.OUTPUT_PATH.as_posix()
    path = score._repo_path(score.OUTPUT_PATH)
    assert path == (score._ROOT / score.OUTPUT_PATH).resolve()
    if path.exists():
        payload = score._read_json(path, "receipt")
        assert payload["package_id"] == score.load_config(verify_package=False)["package_id"]


def test_package_seals_after_finalization() -> None:
    config = score.load_config()
    assert config["score_contract"]["model_scores"] == 72
