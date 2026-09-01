from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_rg_things_six_external_2d_fixed_score_v1 as packet


def test_config_and_sealed_prediction_predecessors_are_valid() -> None:
    config = packet.load_config(verify_package=False)
    packet.validate_config(config)
    prediction_config, prediction_receipt, manifest = packet._load_prediction_evidence(config)
    assert prediction_config["execution_contract"]["candidate_resolution_predictions"] == 120
    assert prediction_receipt["all_solver_gates_pass"] is True
    assert manifest["cell_count"] == 15


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("prediction_binding", "source_cells", 14),
        ("response_contract", "files", 23),
        ("score_contract", "minimum_rg_primary_object_wins_for_strong_signal", 1),
        ("score_contract", "response_parameter_tuning", True),
        ("scientific_boundary", "general_3d_validated", True),
        ("claim_boundary", "publication_ready", True),
    ],
)
def test_material_config_mutations_fail(section: str, key: str, value: object) -> None:
    config = copy.deepcopy(packet.load_config(verify_package=False))
    config[section][key] = value
    with pytest.raises(packet.SixGalaxyScoreError):
        packet.validate_config(config)


def test_preflight_response_inventory_is_exactly_six_by_two_by_two() -> None:
    config = packet.load_config(verify_package=False)
    preflight = packet._load_preflight(config)
    rows = packet._response_rows(preflight)
    assert len(rows) == 24
    assert {object_id for object_id, _resolution, _observable in rows} == set(
        config["response_contract"]["object_order"]
    )
    assert preflight["inventory_counts"]["response_pixels_decoded"] == 0


def test_model_metrics_share_one_sign_and_systemic_velocity() -> None:
    observed = np.asarray([100.0, 110.0, 120.0, 130.0])
    dispersion = np.asarray([2.0, 4.0, 6.0, 8.0])
    prediction = np.asarray([-15.0, -5.0, 5.0, 15.0])
    mask = np.ones(4, dtype=bool)
    metrics = packet._model_metrics(
        observed,
        dispersion,
        prediction,
        mask,
        sign=1.0,
        shared_systemic=115.0,
        minimum_dispersion=3.0,
    )
    assert metrics["rmse_m_s"] == pytest.approx(0.0)
    assert metrics["shared_systemic_velocity_m_s"] == 115.0
    assert metrics["residual_count"] == 4


def _primary_scores(
    config: dict[str, object], rg_winners: set[str], *, aggregate_rg: float
) -> list[dict[str, object]]:
    scores: list[dict[str, object]] = []
    strata = config["response_contract"]["inclination_strata"]
    for object_id, cell_id in config["score_contract"]["primary_cell_by_object"].items():
        rg_loss = aggregate_rg
        comparator = 10.0 if object_id in rg_winners else 0.1
        models = {
            "NEWTON_3D_DST": {"rmse_m_s": comparator},
            "RAR_2016_ON_NEWTON_3D": {"rmse_m_s": comparator + 1.0},
            "MOND_STANDARD_MU_ON_NEWTON_3D": {"rmse_m_s": comparator + 2.0},
            packet._RG: {"rmse_m_s": rg_loss},
        }
        ranking = sorted(packet._CANDIDATES, key=lambda candidate: models[candidate]["rmse_m_s"])
        scores.append(
            {
                "cell_score_id": cell_id,
                "object_id": object_id,
                "inclination_stratum": strata[object_id],
                "models": models,
                "winner": ranking[0],
                "rg_beats_all_three_comparators": object_id in rg_winners,
            }
        )
    return scores


def test_strong_adjudication_requires_four_objects_aggregate_and_all_strata() -> None:
    config = packet.load_config(verify_package=False)
    winners = {"NGC6946", "IC2574", "NGC2841", "DDO154"}
    result = packet._adjudicate_primary(config, _primary_scores(config, winners, aggregate_rg=0.5))
    assert result["strong_external_replication"] is True
    assert result["decision"] == "STRONG_EXTERNAL_SIX_GALAXY_RG_2D_REPLICATION_SIGNAL"


def test_mixed_signal_is_retained_without_strong_overclaim() -> None:
    config = packet.load_config(verify_package=False)
    winners = {"NGC2841", "DDO154"}
    result = packet._adjudicate_primary(config, _primary_scores(config, winners, aggregate_rg=2.0))
    assert result["strong_external_replication"] is False
    assert result["mixed_signal"] is True
    assert result["decision"] == "MIXED_EXTERNAL_RG_2D_SIGNAL_RETAIN_FOR_FOLLOW_UP"


def test_no_signal_is_reported_without_eliminating_the_family() -> None:
    config = packet.load_config(verify_package=False)
    result = packet._adjudicate_primary(config, _primary_scores(config, set(), aggregate_rg=20.0))
    assert result["strong_external_replication"] is False
    assert result["mixed_signal"] is False
    assert result["decision"] == "EXTERNAL_SIX_GALAXY_RG_2D_SIGNAL_NOT_REPLICATED"
    assert config["score_contract"]["retain_every_failure_and_counterexample"] is True


def test_common_mask_requires_identical_candidate_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.asarray([[1.0, np.nan], [3.0, 4.0]])
    dispersion = np.asarray([[2.0, 2.0], [0.0, 2.0]])

    def fake_array(_manifest: object, _cell: str, role: str) -> np.ndarray:
        if role.endswith("eligibility"):
            return np.ones((2, 2), dtype=np.uint8)
        value = np.ones((2, 2), dtype=np.float64)
        if role.startswith("RAR"):
            value[1, 1] = np.nan
        return value

    monkeypatch.setattr(packet, "_load_prediction_array", fake_array)
    mask, arrays = packet._common_mask({}, "CELL", "NATURAL", observed, dispersion)
    assert int(np.count_nonzero(mask)) == 1
    assert set(arrays) == set(packet._CANDIDATES)


def test_atomic_no_clobber_and_conflict(tmp_path: Path) -> None:
    path = tmp_path / "sealed.json"
    assert packet._atomic_no_clobber(path, b"same") == "CREATED"
    assert packet._atomic_no_clobber(path, b"same") == "EXISTING_IDENTICAL"
    with pytest.raises(packet.SixGalaxyScoreError):
        packet._atomic_no_clobber(path, b"different")


def test_score_output_obeys_pre_or_post_run_lifecycle() -> None:
    path = packet._repo_path(packet.OUTPUT_PATH)
    if not path.exists():
        assert packet.status()["status"] == "FROZEN_UNRUN"
        return
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_FIXED_SIX_EXTERNAL_GALAXY_REAL_THINGS_2D_PIXEL_SCORE"
    assert len(receipt["scores"]) == 30
    assert receipt["aggregate"]["all_cells_reported"] is True


def test_config_is_valid_json_and_all_30_cells_are_required() -> None:
    config = json.loads(packet._repo_path(packet.CONFIG_PATH).read_text(encoding="utf-8"))
    assert config["score_contract"]["source_resolution_cells"] == 30
    assert config["score_contract"]["model_scores"] == 120


def test_package_seals_after_finalization() -> None:
    config = packet.load_config()
    assert config["prediction_binding"]["all_solver_gates_pass"] is True
