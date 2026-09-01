from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_rg_sings_seven_holdout_velocity_score_v1 as package


def test_config_freezes_all_cells_four_laws_no_fit_and_no_selection() -> None:
    config = package.load_config()
    assert config["prediction_binding"]["completed_prediction_cells"] == 24
    assert len(config["prediction_binding"]["candidate_ids"]) == 4
    assert config["score_contract"]["all_24_cells_scored"] is True
    assert config["score_contract"]["best_cell_selection_allowed"] is False
    assert config["score_contract"]["parameters_fitted"] == 0
    assert config["score_contract"]["thresholds_tuned"] == 0
    assert config["access_ceiling"]["selection_events"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        ("prediction_binding", "completed_prediction_cells", 23),
        ("geometry_transform", "response_based_geometry_selection", True),
        ("row_gate", "uncertainty_floor_km_s", 5.0),
        ("score_contract", "best_cell_selection_allowed", True),
        ("score_contract", "parameters_fitted", 1),
        ("access_ceiling", "tuning_calls", 1),
        ("claim_boundary", "publication_ready", True),
    ],
)
def test_material_config_mutations_fail_closed(mutation: tuple[str, str, object]) -> None:
    config = package.load_config()
    changed = copy.deepcopy(config)
    changed[mutation[0]][mutation[1]] = mutation[2]
    with pytest.raises(package.VelocityScoreError):
        package.validate_config(changed)


def test_prediction_packet_is_exact_complete_and_blind() -> None:
    config = package.load_config()
    receipt = package._load_prediction_receipt(config)
    assert receipt["completed_prediction_cells"] == 24
    assert receipt["missing_prediction_cells"] == []
    assert all(value == 0 for value in receipt["response_boundary"].values())
    assert len(package._prediction_cell_index(config, receipt)) == 24


def test_response_parsers_match_exact_frozen_counts() -> None:
    config = package.load_config()
    counts = {}
    for object_id in ("NGC2841", "IC2574", "DDO154", "NGC5055", "NGC6946", "NGC7331"):
        binding = package._response_binding(config, object_id)
        counts[object_id] = len(package._parse_sparc(binding))
    binding = package._response_binding(config, "UGC04305")
    counts["UGC04305"] = len(package._parse_little_things(binding))
    assert counts == {
        "NGC2841": 50,
        "IC2574": 34,
        "DDO154": 12,
        "NGC5055": 28,
        "NGC6946": 58,
        "NGC7331": 36,
        "UGC04305": 85,
    }
    assert sum(counts.values()) == 303


def test_geometry_rescaling_is_exact_and_not_response_selected() -> None:
    config = package.load_config()
    rows = package.load_response_rows(config, "IC2574", 53.4)
    raw = package._parse_sparc(package._response_binding(config, "IC2574"))
    expected_radius_scale = 4.0 / 3.91
    expected_velocity_scale = math.sin(math.radians(75.0)) / math.sin(math.radians(53.4))
    assert rows[0]["radius_source_kpc"] == pytest.approx(
        raw[0]["radius_response_kpc"] * expected_radius_scale
    )
    assert rows[0]["velocity_source_km_s"] == pytest.approx(
        raw[0]["velocity_response_km_s"] * expected_velocity_scale
    )
    assert expected_velocity_scale > 1.0


def test_holmberg_ii_all_three_inclinations_are_independently_transformable() -> None:
    config = package.load_config()
    samples = {
        inclination: package.load_response_rows(config, "UGC04305", inclination)[10][
            "velocity_source_km_s"
        ]
        for inclination in (27.0, 38.0, 49.0)
    }
    assert samples[27.0] > samples[38.0] > samples[49.0]


def test_prediction_interpolation_respects_frozen_mask() -> None:
    config = package.load_config()
    receipt = package._load_prediction_receipt(config)
    index = package._prediction_cell_index(config, receipt)
    _source, cell = index["NGC2841__IRAC1_FIXED_ML0P6__I73P7"]
    value, reason = package._interpolated_prediction(cell, 5.0, "NEWTON_3D_DST")
    assert reason is None
    assert value is not None and value > 0.0
    value, reason = package._interpolated_prediction(cell, 100.0, "NEWTON_3D_DST")
    assert value is None
    assert reason == "RADIUS_OUTSIDE_PREDICTION_GRID"


def test_full_score_recomputes_all_cells_rows_and_counterexamples() -> None:
    config = package.load_config()
    receipt = package.build_receipt(config)
    assert receipt["status"] == "PASS_ONE_PASS_ALL_CELL_SCORE_RETAINING_EVERY_COUNTEREXAMPLE"
    assert receipt["response_files_opened"] == 7
    assert receipt["response_rows_parsed"] == 303
    assert receipt["prediction_cells_opened"] == 24
    assert receipt["scored_cell_count"] == 24
    assert len(receipt["cell_scores"]) == 24
    assert len(receipt["primary_aggregate"]["primary_cell_ids"]) == 7
    assert set(receipt["counterexample_cells"]).issubset(
        {row["cell_run_id"] for row in receipt["cell_scores"]}
    )
    assert receipt["claim_boundary"]["unique_theory_established"] is False
    assert receipt["claim_boundary"]["publication_ready"] is False


def test_receipt_forgery_fails_closed() -> None:
    config = package.load_config()
    receipt = package.build_receipt(config)
    changed = copy.deepcopy(receipt)
    changed["claim_boundary"]["publication_ready"] = True
    changed["content_sha256"] = package.content_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(package.VelocityScoreError, match="overpromoted"):
        package.validate_receipt(config, changed)


def test_atomic_no_clobber_is_exact(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert package._atomic_no_clobber(output, b"one") == "CREATED"
    assert package._atomic_no_clobber(output, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(package.VelocityScoreError, match="differs"):
        package._atomic_no_clobber(output, b"two")
    assert output.read_bytes() == b"one"


def test_package_seals_match_current_files() -> None:
    if package._CONFIG_RAW_SHA256 == "0" * 64:
        pytest.skip("package pins not yet sealed")
    assert (
        package.file_sha256(package._repo_path(package.CONFIG_PATH)) == package._CONFIG_RAW_SHA256
    )
    assert package.content_sha256(package.load_config()) == package._CONFIG_CONTENT_SHA256
    assert (
        package.module_semantic_sha256(package._repo_path(package.MODULE_PATH))
        == package._MODULE_SEMANTIC_SHA256
    )
    assert package.file_sha256(package._repo_path(package.TEST_PATH)) == package._TEST_RAW_SHA256
