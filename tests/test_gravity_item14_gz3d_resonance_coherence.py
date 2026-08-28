from __future__ import annotations

import copy
import gzip
import inspect
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from sigma_theory_compiler import gravity_item14_gz3d_resonance_coherence as coherence

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_real_masks_fresh_response_and_claim_boundaries() -> None:
    config = coherence.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 14
    assert config["predecessor"]["required_decision"] == (
        "REJECT_ITEM13_DISTURBANCE_RETAIN_AGE_LEAD_ADVANCE_ITEM14"
    )
    metadata = config["sources"]["gz3d_metadata"]
    assert metadata["observed_rows"] == 29813
    assert metadata["file_bytes"] == 7551360
    assert metadata["file_sha256"] == (
        "5b88b7aa65f99f5e1bda0eacecb7a26288dac5c5347029cf4f48fa9a120e80fd"
    )
    assert config["sources"]["prefreeze_access"]["metadata_row_values_read"] == 0
    assert config["sources"]["prefreeze_access"]["mask_pixel_values_read"] == 0
    assert config["sources"]["prefreeze_access"]["maps_payload_downloads"] == 0
    assert config["sources"]["prefreeze_access"]["maps_pixel_values_read"] == 0
    assert config["sources"]["prefreeze_access"]["resolved_kinematic_response_objects_read"] == 0
    assert config["sources"]["response"]["daptype"] == "HYB10-MILESHC-MASTARSSP"
    assert config["scientific_contract"]["primary_response"].startswith(
        "log10 of the robust line-of-sight stellar-velocity span"
    )
    assert config["sample"]["maximum_total_objects"] == 320
    assert config["sample"]["exploration_objects"] == 240
    assert config["sample"]["confirmation_objects"] == 80
    assert config["candidate_generator"]["candidate_cells"] == 262144
    assert len(config["candidate_generator"]["families"]) == 12
    assert config["sources"]["response"]["confirmation_query_forbidden"] is True
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["authorization"]["response_query_allowed_before_sample_freeze"] is False
    assert config["authorization"]["confirmation_response_query_allowed"] is False
    assert config["authorization"]["post_response_candidate_generation_allowed"] is False
    assert config["quality"]["minimum_halpha_equivalent_width_angstrom"] == 3.0
    assert config["quality"]["minimum_quality_passing_per_outer_fold"] == 20
    assert config["quality"]["minimum_quality_passing_per_gate_stratum"] == 20
    assert config["mask_feature_extraction"]["expected_shape_pixels"] == [525, 525]
    assert all(value is False for value in config["claim_boundaries"].values())


def _synthetic_masks(transform: str = "identity") -> bytes:
    size = 129
    yy, xx = np.indices((size, size), dtype=np.float64)
    xx -= (size - 1) / 2
    yy -= (size - 1) / 2
    radius = np.hypot(xx, yy)
    angle = np.arctan2(yy, xx)
    target = 0.55 * np.log(np.maximum(radius, 4.0) / 4.0)
    wrapped = np.angle(np.exp(2j * (angle - target))) / 2.0
    spiral = ((radius >= 5) & (radius <= 62) & (np.abs(wrapped) < 0.13)).astype(float)
    bar = ((np.abs(xx) <= 22) & (np.abs(yy) <= 2.5)).astype(float)
    if transform == "rotate":
        spiral = np.rot90(spiral)
        bar = np.rot90(bar)
    elif transform == "reflect":
        spiral = np.fliplr(spiral)
        bar = np.fliplr(bar)
    buffer = io.BytesIO()
    fits.HDUList(
        [
            fits.PrimaryHDU(np.zeros_like(spiral)),
            fits.ImageHDU(np.zeros_like(spiral)),
            fits.ImageHDU(np.zeros_like(spiral)),
            fits.ImageHDU(spiral),
            fits.ImageHDU(bar),
        ]
    ).writeto(buffer)
    return gzip.compress(buffer.getvalue(), mtime=0)


def _predictor() -> dict[str, object]:
    return {"log_half_light_radius": math.log10(3.0)}


def _synthetic_config() -> dict[str, object]:
    config = copy.deepcopy(coherence.load_config(ROOT))
    config["mask_feature_extraction"]["expected_shape_pixels"] = [129, 129]
    return config


def _synthetic_maps_response() -> bytes:
    size = 41
    yy, xx = np.indices((size, size), dtype=np.float64)
    xx -= (size - 1) / 2
    yy -= (size - 1) / 2
    radius = np.hypot(xx, yy) / 12.0
    azimuth = np.mod(np.degrees(np.arctan2(yy, xx)), 360.0)
    amplitude = np.where(radius < 0.8, 90.0, 120.0)
    stellar_velocity = amplitude * np.cos(np.radians(azimuth))
    halpha_velocity = 1.05 * amplitude * np.cos(np.radians(azimuth))
    shape = stellar_velocity.shape

    def image(name: str, data: np.ndarray, channels: list[str] | None = None) -> fits.ImageHDU:
        hdu = fits.ImageHDU(data=data, name=name)
        for ordinal, channel in enumerate(channels or [], start=1):
            hdu.header[f"C{ordinal}"] = channel
        return hdu

    primary = fits.PrimaryHDU()
    primary.header["VERSDRP3"] = "v3_1_1"
    primary.header["VERSDAP"] = "3.1.0"
    primary.header["DAPTYPE"] = "HYB10-MILESHC-MASTARSSP"
    primary.header["DAPFRMT"] = "MAPS"
    primary.header["PLATEIFU"] = "9999-1901"
    primary.header["MANGAID"] = "1-999"
    primary.header["DRP3QUAL"] = 0
    primary.header["DAPQUAL"] = 0
    spx_ellcoo = np.stack((radius * 10.0, radius, radius * 2.0, azimuth))
    bin_lwellcoo = spx_ellcoo.copy()
    bin_ids = np.arange(size * size, dtype=np.int32).reshape(shape)
    binid_cube = np.stack([bin_ids] * 5)
    stellar_fom = np.ones((9, *shape), dtype=np.float64)
    emission_shape = (35, *shape)
    emission_velocity = np.zeros(emission_shape, dtype=np.float64)
    emission_velocity[23] = halpha_velocity
    emission_ivar = np.ones(emission_shape, dtype=np.float64)
    emission_mask = np.zeros(emission_shape, dtype=np.int32)
    emission_anr = np.full(emission_shape, 10.0, dtype=np.float64)
    emission_ew = np.full(emission_shape, 5.0, dtype=np.float64)
    emission_lfom = np.ones(emission_shape, dtype=np.float64)
    line_channels = [f"line-{index + 1}" for index in range(35)]
    line_channels[23] = "Ha-6564"
    hdus = fits.HDUList(
        [
            primary,
            image(
                "SPX_ELLCOO",
                spx_ellcoo,
                ["Elliptical radius", "R/Re", "R h/kpc", "Elliptical azimuth"],
            ),
            image(
                "BIN_LWELLCOO",
                bin_lwellcoo,
                [
                    "Lum. weighted elliptical radius",
                    "R/Re",
                    "R h/kpc",
                    "Lum. weighted elliptical azimuth",
                ],
            ),
            image(
                "BINID",
                binid_cube,
                [
                    "Binned spectra",
                    "Stellar continua",
                    "Em. line moments",
                    "Em. line models",
                    "Spectral indices",
                ],
            ),
            image("BIN_SNR", np.full(shape, 10.0)),
            image("STELLAR_VEL", stellar_velocity),
            image("STELLAR_VEL_IVAR", np.ones(shape)),
            image("STELLAR_VEL_MASK", np.zeros(shape, dtype=np.int32)),
            image(
                "STELLAR_FOM",
                stellar_fom,
                ["rms", "frms", "rchi2", "p4", "p5", "p6", "p7", "p8", "p9"],
            ),
            image("EMLINE_GVEL", emission_velocity, line_channels),
            image("EMLINE_GVEL_IVAR", emission_ivar, line_channels),
            image("EMLINE_GVEL_MASK", emission_mask, line_channels),
            image("EMLINE_GANR", emission_anr, line_channels),
            image("EMLINE_GEW", emission_ew, line_channels),
            image("EMLINE_GEW_MASK", emission_mask.copy(), line_channels),
            image("EMLINE_LFOM", emission_lfom, line_channels),
        ]
    )
    buffer = io.BytesIO()
    hdus.writeto(buffer, checksum=True)
    return gzip.compress(buffer.getvalue(), mtime=0)


def test_mask_features_are_finite_and_rotation_reflection_invariant() -> None:
    config = _synthetic_config()
    original = coherence.derive_mask_features(_synthetic_masks(), _predictor(), config)
    rotated = coherence.derive_mask_features(_synthetic_masks("rotate"), _predictor(), config)
    reflected = coherence.derive_mask_features(_synthetic_masks("reflect"), _predictor(), config)
    assert original["spiral_nonzero_pixels"] > 25
    assert all(np.isfinite(float(value)) for value in original.values())
    for key in config["mask_feature_extraction"]["features"]:
        assert float(rotated[key]) == pytest.approx(float(original[key]), abs=1e-9)
        assert float(reflected[key]) == pytest.approx(float(original[key]), abs=1e-9)


def test_mask_quality_rejects_empty_spiral() -> None:
    config = _synthetic_config()
    empty = io.BytesIO()
    zeros = np.zeros((129, 129))
    fits.HDUList(
        [
            fits.PrimaryHDU(zeros),
            fits.ImageHDU(zeros),
            fits.ImageHDU(zeros),
            fits.ImageHDU(zeros),
            fits.ImageHDU(zeros),
        ]
    ).writeto(empty)
    with pytest.raises(coherence.GravityItem14CoherenceError, match="spiral"):
        coherence.derive_mask_features(
            gzip.compress(empty.getvalue(), mtime=0), _predictor(), config
        )


def test_mask_quality_rejects_wrong_image_geometry() -> None:
    config = coherence.load_config(ROOT)
    with pytest.raises(coherence.GravityItem14CoherenceError, match="shape"):
        coherence.derive_mask_features(_synthetic_masks(), _predictor(), config)


def test_resolved_maps_response_measures_outer_inner_ratios() -> None:
    result = coherence.derive_radial_response(
        _synthetic_maps_response(),
        {"plateifu": "9999-1901", "mangaid": "1-999"},
        coherence.load_config(ROOT),
    )
    assert result["stellar_inner_measurements"] >= 8
    assert result["stellar_outer_measurements"] >= 8
    assert result["halpha_inner_measurements"] >= 20
    assert result["halpha_outer_measurements"] >= 20
    assert float(result["stellar_outer_to_inner_span_ratio"]) > 1.2
    assert float(result["halpha_outer_to_inner_span_ratio"]) > 1.2


def test_stellar_map_measurements_are_deduplicated_by_bin() -> None:
    bin_ids = np.asarray([[1, 1], [2, 2]])
    valid = np.ones((2, 2), dtype=bool)
    inverse_variance = np.asarray([[1.0, 2.0], [4.0, 3.0]])
    velocity = np.asarray([[10.0, 11.0], [20.0, 21.0]])
    radius = np.asarray([[0.5, 0.5], [1.0, 1.0]])
    azimuth = np.asarray([[0.0, 0.0], [180.0, 180.0]])
    values, radii, angles = coherence._unique_stellar_measurements(
        bin_ids, valid, inverse_variance, velocity, radius, azimuth
    )
    assert values.tolist() == [11.0, 20.0]
    assert radii.tolist() == [0.5, 1.0]
    assert angles.tolist() == [0.0, 180.0]


@pytest.mark.parametrize(
    ("value", "valid"),
    [(0, True), (12, True), (-1, False), (2.5, False), (float("nan"), False)],
)
def test_vote_count_validation(value: float, valid: bool) -> None:
    if valid:
        assert coherence._nonnegative_integer_count(value, "test count") == int(value)
    else:
        with pytest.raises(coherence.GravityItem14CoherenceError, match="test count"):
            coherence._nonnegative_integer_count(value, "test count")


def test_access_requires_exact_commit_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coherence, "SCIENTIFIC_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(coherence.GravityItem14CoherenceError, match="not bound"):
        coherence.write_prepared_sources(ROOT)
    monkeypatch.setattr(coherence, "SAMPLE_FREEZE_COMMIT", "PENDING_TEST")
    with pytest.raises(coherence.GravityItem14CoherenceError, match="not bound"):
        coherence.write_response_source(ROOT)


def _sample_candidate(ordinal: int, cell: str) -> dict[str, object]:
    bar_state, mass_state = cell.split("|")
    return {
        "mangaid": f"1-{ordinal}",
        "plateifu": f"{ordinal}-1",
        "ra": "150",
        "dec": "2",
        "gz3d_file_name": f"gz3d_1-{ordinal}_37_{ordinal}.fits.gz",
        "gz3d_official_sha1": "0" * 40,
        "sample_cell": cell,
        "bar_vote_state": bar_state,
        "stellar_mass_state": mass_state,
        "log_half_light_radius": _predictor()["log_half_light_radius"],
        "log_stellar_mass": "10.5",
        "log_surface_density": "8.5",
        "axis_ratio": "0.7",
        "sersic_index": "2",
        "g_minus_r_color": "0.7",
        "redshift": "0.03",
        "log_surface_brightness": "0",
        "log_snr": "1",
        "ttype_normalized": "0",
        "bar_strength": "0.2",
        "edge_on": 0,
        "concentration_normalized": "0",
        "spiral_vote_fraction": "0.5",
        "bar_vote_fraction": "0.3" if bar_state == "bar_high" else "0.1",
        "prior_age_lead": "0.1",
    }


def test_target_blind_sample_selects_balanced_cells_before_response() -> None:
    config = _synthetic_config()
    config["sample"]["objects_per_cell"] = 4
    config["sample"]["exploration_objects"] = 12
    config["sample"]["confirmation_objects"] = 4
    config["sample"]["maximum_total_objects"] = 16
    cells = [
        f"{bar}|{mass}" for bar in ("bar_low", "bar_high") for mass in ("lower_mass", "higher_mass")
    ]
    candidates = []
    ordinal = 0
    for cell in cells:
        for _ in range(6):
            ordinal += 1
            candidates.append(_sample_candidate(ordinal, cell))

    def loader(root: Path, row: dict[str, object], value: dict[str, object]) -> tuple[bytes, str]:
        return _synthetic_masks(), "a" * 64

    objects, features, failures = coherence._select_with_mask_features(
        ROOT, candidates, config, loader
    )
    assert len(objects) == 16
    assert len(features) == 16
    assert failures == []
    assert sum(row["role"] == "exploration" for row in objects) == 12
    assert sum(row["role"] == "reserved_confirmation" for row in objects) == 4
    assert all(row["response_read"] is False for row in objects)
    assert {cell: sum(row["sample_cell"] == cell for row in objects) for cell in cells} == {
        cell: 4 for cell in cells
    }


def test_pseudorandom_generator_is_deterministic_and_fully_labeled() -> None:
    config = coherence.load_config(ROOT)
    first = coherence.generate_candidates(config)
    second = coherence.generate_candidates(config)
    assert coherence._candidate_digest(first) == coherence._candidate_digest(second)
    assert len(first["family"]) == 262144
    assert all(np.array_equal(first[key], second[key]) for key in first)
    labels = {row["origin_status"] for row in config["candidate_generator"]["families"]}
    assert labels == {
        "KNOWN_FORMULA_TRANSFORM",
        "KNOWN_FAMILY_COMBINATION",
        "COMBINATION",
        "UNRESOLVED",
    }


def test_all_coherence_families_are_finite_and_distinct() -> None:
    arrays = {
        "family": np.arange(12, dtype=np.int16),
        "threshold": np.linspace(-1.1, 1.1, 12),
        "scale": np.linspace(0.2, 1.3, 12),
        "power": np.linspace(0.5, 3.0, 12),
        "phase": np.linspace(0.1, 2.0, 12),
        "modulation": np.zeros(12, dtype=np.int8),
    }
    count = 19
    random = np.random.default_rng(14)
    data = {
        key: random.normal(size=count)
        for key in (
            "mode_m1",
            "mode_m2",
            "mode_m3",
            "mode_m4",
            "mode_entropy",
            "phase_linearity",
            "pitch_abs",
            "phase_twist",
            "bar_phase_lock",
            "bar_radius_ratio",
            "coverage",
            "bar_vote_modulation",
            "surface_modulation",
            "mass_modulation",
            "age_modulation",
            "coverage_modulation",
        )
    }
    components = coherence._candidate_components(arrays, data, 0, 12, np)
    assert components.shape == (12, count)
    assert np.all(np.isfinite(components))
    assert len({np.round(row, 10).tobytes() for row in components}) == 12


def test_nested_selector_executes_primary_and_honest_secondary() -> None:
    config = copy.deepcopy(coherence.load_config(ROOT))
    config["candidate_generator"]["candidate_cells"] = 96
    config["evaluation"]["candidate_batch_size"] = 24
    config["evaluation"]["cpu_crosscheck_candidates"] = 8
    count = 25
    random = np.random.default_rng(1400)
    data = {
        "folds": np.arange(count) % 5,
        "y": random.normal(0, 0.15, count),
        "y_halpha": random.normal(2.4, 0.15, count),
        "design_control": random.normal(size=(count, 16)),
        "design_secondary": random.normal(size=(count, 16)),
    }
    for key in (
        "mode_m1",
        "mode_m2",
        "mode_m3",
        "mode_m4",
        "mode_entropy",
        "phase_linearity",
        "pitch_abs",
        "phase_twist",
        "bar_phase_lock",
        "bar_radius_ratio",
        "coverage",
        "bar_vote_modulation",
        "surface_modulation",
        "mass_modulation",
        "age_modulation",
        "coverage_modulation",
    ):
        data[key] = random.normal(size=count)
    predictions, selections, compute = coherence._nested_select(data, config)
    assert set(predictions) == {
        "control",
        "full",
        "secondary_control",
        "secondary_full",
    }
    assert all(np.all(np.isfinite(value)) for value in predictions.values())
    assert len(selections) == 5
    assert compute["candidate_cells"] == 96
    assert compute["candidate_galaxy_score_evaluations"] == 96 * count * 20


def test_formula_and_sample_builders_have_no_response_parameter() -> None:
    for builder in (
        coherence.derive_mask_features,
        coherence.generate_candidates,
        coherence._select_with_mask_features,
    ):
        signature = " ".join(inspect.signature(builder).parameters).lower()
        assert "velocity" not in signature
        assert "response" not in signature
    component_source = inspect.getsource(coherence._candidate_components)
    assert "median" not in component_source
    assert "std" not in component_source
    response_source = inspect.getsource(coherence.write_response_source)
    assert 'row["role"] == "exploration"' in response_source
    assert 'row["role"] == "reserved_confirmation"' in response_source
    preparation_source = inspect.getsource(coherence.write_prepared_sources)
    assert "_maps_payload" not in preparation_source
    assert "derive_radial_response" not in preparation_source
    filename, url = coherence._maps_location(coherence.load_config(ROOT), "7443-12703")
    assert filename == "manga-7443-12703-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
    assert url.endswith("/HYB10-MILESHC-MASTARSSP/7443/12703/" + filename)


def test_response_receipt_requires_both_commit_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = {
        "objects": [
            {"plateifu": "1-1", "role": "exploration"},
            {"plateifu": "2-2", "role": "reserved_confirmation"},
        ]
    }
    monkeypatch.setattr(coherence, "_load_prepared", lambda root: (sample, {}, {}))
    source = coherence._content_hashed(
        {
            "scientific_freeze_commit": coherence.SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": coherence.SAMPLE_FREEZE_COMMIT,
            "records": [{"plateifu": "1-1"}],
            "failures": [],
            "files": [{"plateifu": "1-1", "fits_checksum_verified": True}],
            "counts": {
                "exploration_response_objects_attempted": 1,
                "exploration_response_objects_parsed": 1,
                "exploration_response_failures": 0,
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": {"confirmation_opened": False},
        }
    )
    coherence.validate_response_source(source, ROOT)
    wrong = coherence._content_hashed(
        {
            **{key: value for key, value in source.items() if key != "content_sha256"},
            "scientific_freeze_commit": "wrong",
        }
    )
    with pytest.raises(coherence.GravityItem14CoherenceError, match="scientific"):
        coherence.validate_response_source(wrong, ROOT)


def test_stored_artifacts_replay_if_present() -> None:
    config = coherence.load_config(ROOT)
    prepared_paths = [
        ROOT / config["outputs"][key]
        for key in ("sample_manifest", "mask_feature_source", "candidate_manifest")
    ]
    if all(path.exists() for path in prepared_paths):
        values = [json.loads(path.read_text(encoding="utf-8")) for path in prepared_paths]
        coherence.validate_prepared_sources(*values, ROOT)
    response_path = ROOT / config["outputs"]["response_source"]
    if response_path.exists():
        coherence.validate_response_source(
            json.loads(response_path.read_text(encoding="utf-8")), ROOT
        )


def test_stored_result_replays_if_present() -> None:
    config = coherence.load_config(ROOT)
    path = ROOT / config["outputs"]["result"]
    if not path.exists():
        pytest.skip("GZ3D resonance/coherence exploration has not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    coherence.validate_receipt(stored, ROOT)
    coherence.check_receipt(ROOT)
    assert stored["counts"]["candidate_cells"] == 262144
    assert stored["counts"]["confirmation_response_rows"] == 0
    assert stored["counts"]["post_response_formula_cells"] == 0
    assert stored["counts"]["paid_model_calls"] == 0
