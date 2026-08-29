from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_strata_development_scoring as scoring

CONFIG_SHA256 = "92e6ec7e1b7d5d4ed8d2ff9a23d4e5a3ff307ac4d82360a4becc2e709f8d4511"


def config() -> dict:
    return scoring.load_config(scoring.ROOT / scoring.CONFIG_PATH, CONFIG_SHA256)


def analysis() -> dict:
    return scoring.build_analysis(config())


def test_config_and_all_predecessor_seals_are_exact() -> None:
    frozen = config()
    assert frozen["chronology"]["predictor_strata_commit"] == (
        "b6c13fd24007f1ad92a6bd46456834092b55f20b"
    )
    assert frozen["inputs"] == scoring.EXPECTED_INPUTS
    assert frozen["model_freeze"] == scoring.EXPECTED_MODEL_FREEZE
    scoring.validate_inputs(frozen)


def test_all_eight_objects_and_both_development_splits_are_reported() -> None:
    result = analysis()
    assert [row["cluster_id"] for row in result["per_object"]] == scoring.CLUSTERS
    assert all(set(row["scores"]) == set(scoring.SPLITS) for row in result["per_object"])
    assert (
        sum(
            row["scores"][split]["rows"] for row in result["per_object"] for split in scoring.SPLITS
        )
        == 54
    )


def test_equal_cluster_primary_aggregates_reproduce_frozen_covariance_pilot() -> None:
    primary = analysis()["whole_population"]["development_holdout"]["full_covariance"]
    assert primary["candidate_score_equal_cluster_mean"] == pytest.approx(3.8629233674315238)
    assert primary["nfw_score_equal_cluster_mean"] == pytest.approx(12.62010166320426)
    assert primary["candidate_advantage_equal_cluster_mean"] == pytest.approx(8.757178295772736)
    assert primary["candidate_wins"] == 4


def test_covariance_pressure_flips_are_exact_and_not_hidden() -> None:
    result = analysis()["whole_population"]
    assert result["development_holdout"]["positive_diagonal_to_negative_full_clusters"] == [
        "A85",
        "ZW1215",
    ]
    assert result["development_train"]["positive_diagonal_to_negative_full_clusters"] == ["ZW1215"]
    assert result["development_holdout"]["negative_diagonal_to_positive_full_clusters"] == []


def test_predictor_partitions_and_unclassified_semantics_are_exact() -> None:
    partitions = analysis()["partition_results"]
    assert partitions["relaxation_proxy"]["group_a"] == ["A1795", "A2142", "A85", "ZW1215"]
    assert partitions["cool_core"]["group_a"] == ["A1644", "A1795", "A85"]
    assert partitions["stellar_profile_availability"]["group_a"] == [
        "A1795",
        "A2142",
        "A2319",
        "A85",
        "ZW1215",
    ]
    assembly = partitions["positive_assembly_vs_unclassified"]
    assert assembly["group_a"] == ["A2142", "A2319", "A3266", "A85"]
    assert assembly["group_b_label"] == "explicitly_unclassified_not_negative"
    assert assembly["group_b"] == ["A1644", "A1795", "A2255", "ZW1215"]


def test_exact_permutations_have_all_finite_assignments_and_holm_adjustment() -> None:
    result = analysis()
    expected_enumerations = {
        "relaxation_proxy": 70,
        "cool_core": 56,
        "stellar_profile_availability": 56,
        "positive_assembly_vs_unclassified": 70,
    }
    for name, expected in expected_enumerations.items():
        cell = result["partition_results"][name]
        assert (
            cell["splits"]["development_holdout"]["exact_permutation"]["candidate_advantage"][
                "enumerations"
            ]
            == expected
        )
        assert cell["primary_flip_enrichment_exact_permutation"]["enumerations"] == expected
        assert 0 <= result["multiplicity"]["candidate_advantage_holm_p"][name] <= 1


def test_A3266_boundary_cell_is_singleton_descriptive_only() -> None:
    boundary = analysis()["partition_results"]["boundary_method"]
    assert boundary["group_b"] == ["A3266"]
    assert boundary["inferential"] is False
    assert "exact_permutation" not in boundary["splits"]["development_holdout"]
    assert "primary_flip_enrichment_exact_permutation" not in boundary


def test_frozen_gates_fail_without_single_object_veto() -> None:
    gates = analysis()["gates"]
    assert gates["candidate_absolute_primary"]["passed"] is False
    assert gates["candidate_vs_nfw_primary"]["passed"] is False
    assert gates["covariance_flip_explained_by_any_frozen_stratum"]["passed"] is False
    assert config()["gates"]["single_counterexample_is_universal_veto"] is False


def test_receipt_is_exact_fail_closed_and_claim_ceiling_is_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = config()
    expected = scoring.expected_receipt(frozen)
    assert expected["readiness"]["CP5_13_task_complete"] is False
    assert expected["claim_boundary"]["scientific_claim_allowed"] is False
    assert expected["compute_and_access_accounting"]["new_raw_target_rows_opened"] == 0
    receipt = scoring.ROOT / scoring.RECEIPT_PATH
    if receipt.exists():
        assert json.loads(receipt.read_text(encoding="utf-8")) == expected
    forged = dict(expected)
    forged["decision"] = "PASS"
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    original_confined = scoring.confined
    monkeypatch.setattr(
        scoring,
        "confined",
        lambda path: path.resolve() if path == forged_path else original_confined(path),
    )
    with pytest.raises(RuntimeError, match="receipt changed"):
        scoring.check(scoring.ROOT / scoring.CONFIG_PATH, CONFIG_SHA256, forged_path)


def test_wrong_config_seal_and_no_clobber_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="config SHA256 mismatch"):
        scoring.load_config(scoring.ROOT / scoring.CONFIG_PATH, "0" * 64)
    target = tmp_path / "existing.json"
    target.write_text("original", encoding="utf-8")
    monkeypatch.setattr(scoring, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        scoring.write_json_no_clobber(target, {"replacement": True})
    assert target.read_text(encoding="utf-8") == "original"
