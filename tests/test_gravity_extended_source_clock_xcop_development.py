from __future__ import annotations

import json
from pathlib import Path

import pytest

import sigma_theory_compiler.gravity_extended_source_clock_xcop_development as clock

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "runs/gravity/theory/extended-source-clock-xcop-development-v1.json"


def _result() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_config_and_authorization_are_exact() -> None:
    config = clock.load_config(ROOT)
    authorization = clock.validate_authorization(
        ROOT, config, ROOT / config["authorization"]["path"]
    )
    assert authorization["authorized"] is True
    assert authorization["authorized_by"] == "Henry"
    assert authorization["approval_phrase"] == config["authorization"]["required_phrase"]


def test_result_rebuilds_without_target_access() -> None:
    status = clock.check_result(ROOT)
    assert status == {
        "valid": True,
        "decision": "EXTENDED_SOURCE_CLOCK_DOES_NOT_RANK_FIRST_ON_FROZEN_DEVELOPMENT_HOLDOUT",
        "clock_rank": 2,
        "holdout_clock_score": 136.45873660378902,
        "confirmation_rows_opened": 0,
    }


def test_only_eight_development_clusters_were_opened() -> None:
    result = _result()
    clusters = sorted({row["cluster"] for row in result["input_file_ledger"]})
    assert clusters == sorted(
        ["A1644", "A1795", "A2142", "A2255", "A2319", "A3266", "A85", "ZW1215"]
    )
    assert not {"A2029", "A3158", "A644", "RXC1825"} & set(clusters)
    assert result["access_and_compute"]["unique_development_files_opened"] == 29
    assert result["access_and_compute"]["development_file_bytes_opened"] == 538560


def test_forbidden_access_and_tuning_remain_zero() -> None:
    access = _result()["access_and_compute"]
    for key in (
        "confirmation_files_opened",
        "confirmation_rows_opened",
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        assert access[key] == 0


def test_holdout_ranking_and_scores_are_frozen() -> None:
    scoring = _result()["scoring"]
    assert scoring["primary_holdout_ranking"] == [
        "previous_cross_scale_candidate",
        "extended_source_clock",
        "empirical_rar",
        "newtonian_baryons",
    ]
    scores = scoring["scores"]["development_holdout"]
    assert scores["previous_cross_scale_candidate"]["score"] == pytest.approx(15.909111282160886)
    assert scores["extended_source_clock"]["score"] == pytest.approx(136.45873660378902)
    assert scores["empirical_rar"]["score"] == pytest.approx(148.04264785509645)
    assert scores["newtonian_baryons"]["score"] == pytest.approx(807.5612830912955)


def test_clock_improves_over_rar_and_newtonian_but_not_previous_candidate() -> None:
    comparisons = _result()["scoring"]["comparisons"]["development_holdout"]
    assert comparisons["empirical_rar"]["fractional_improvement"] == pytest.approx(
        0.07824712283345348
    )
    assert comparisons["newtonian_baryons"]["fractional_improvement"] == pytest.approx(
        0.8310236765172381
    )
    assert comparisons["previous_cross_scale_candidate"]["fractional_improvement"] == pytest.approx(
        -7.5773953166574515
    )


def test_every_counterexample_is_retained() -> None:
    comparisons = _result()["scoring"]["comparisons"]["development_holdout"]
    assert comparisons["empirical_rar"]["cluster_counterexamples"] == ["A1644"]
    assert comparisons["newtonian_baryons"]["cluster_counterexamples"] == []
    assert comparisons["previous_cross_scale_candidate"]["cluster_counterexamples"] == [
        "A1644",
        "A1795",
        "A2142",
        "A2255",
        "A2319",
        "A3266",
        "A85",
        "ZW1215",
    ]


def test_claim_ceiling_remains_development_only() -> None:
    claims = _result()["claims"]
    assert claims["development_only_real_cluster_evidence_completed"] is True
    assert claims["confirmation_evidence_completed"] is False
    assert claims["covariant_time_theory_established"] is False
    assert claims["same_action_lensing_established"] is False
    assert claims["alternative_to_gr_established"] is False
    assert claims["dark_matter_eliminated"] is False
    assert claims["scientific_claim_allowed"] is False
