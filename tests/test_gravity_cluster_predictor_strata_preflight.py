from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_cluster_predictor_strata_preflight as preflight,
)

CONFIG_SHA256 = "947d38b1e083b75b1f1b7c5e1f7250e3f3b1541a8e312708abb1e907a97b2144"


def config() -> dict:
    return preflight.load_config(preflight.ROOT / preflight.CONFIG_PATH, CONFIG_SHA256)


def raw_contract() -> dict:
    value = config()
    value.pop("_config_sha256")
    return value


def strata() -> dict:
    return preflight.build_strata(config())


def test_population_aliases_provenance_and_public_source_seals_are_frozen() -> None:
    frozen = config()
    assert frozen["population"]["cluster_ids"] == preflight.CLUSTERS
    assert frozen["population"]["replacement_allowed"] is False
    assert frozen["local_population_binding"] == {
        "path": "configs/gravity_item59_xcop_forward_observable_gate_v1.json",
        "file_sha256": ("7d17fa71dce7bfb44ab517a47b1cbe0ed17826cc2ffe36f871296b4d8632878b"),
    }
    assert [source["arxiv_id"] for source in frozen["public_sources"]] == [
        "2303.15102",
        "1805.00042",
    ]
    assert [source["doi"] for source in frozen["public_sources"]] == [
        "10.1051/0004-6361/202245779",
        "10.1051/0004-6361/201833325",
    ]
    assert all(len(source["payload_sha256"]) == 64 for source in frozen["public_sources"])
    assert all(row["aliases"][0] == row["cluster_id"] for row in frozen["cluster_metadata"])
    assert frozen["cluster_metadata"][-1]["aliases"] == ["ZW1215", "ZwCl1215"]


def test_published_predictors_and_cool_core_rule_are_exact() -> None:
    frozen = config()
    rows = {row["cluster_id"]: row for row in frozen["cluster_metadata"]}
    assert {cluster_id: row["central_entropy_kev_cm2"] for cluster_id, row in rows.items()} == {
        "A1644": 19.0,
        "A1795": 19.0,
        "A2142": 68.1,
        "A2255": 529.1,
        "A2319": 270.2,
        "A3266": 72.5,
        "A85": 12.5,
        "ZW1215": 163.2,
    }
    assert {cluster_id: row["published_cool_core_class"] for cluster_id, row in rows.items()} == {
        "A1644": "CC",
        "A1795": "CC",
        "A2142": "NCC",
        "A2255": "NCC",
        "A2319": "NCC",
        "A3266": "NCC",
        "A85": "CC",
        "ZW1215": "NCC",
    }
    assert rows["A2319"]["morphology"]["centroid_shift_x1e3"] == {
        "value": 33.069,
        "minus": 0.176,
        "plus": 0.179,
    }


def test_response_blind_medians_scores_and_strata_are_exact() -> None:
    result = strata()
    assert result["frozen_morphology_medians"] == pytest.approx(
        {
            "concentration_csb": 0.21,
            "centroid_shift_x1e3": 9.1355,
            "gini": 0.701,
            "zernike_cz": 0.594,
        }
    )
    assert result["frozen_relaxation_proxy_median"] == pytest.approx(0.5)
    rows = {row["cluster_id"]: row for row in result["cluster_rows"]}
    expected = {
        "A1644": (0.75, "disturbed_proxy"),
        "A1795": (0.10714285714285714, "relaxed_proxy"),
        "A2142": (0.2857142857142857, "relaxed_proxy"),
        "A2255": (0.6785714285714286, "disturbed_proxy"),
        "A2319": (0.7857142857142857, "disturbed_proxy"),
        "A3266": (0.7857142857142858, "disturbed_proxy"),
        "A85": (0.3214285714285714, "relaxed_proxy"),
        "ZW1215": (0.2857142857142857, "relaxed_proxy"),
    }
    for cluster_id, (score, label) in expected.items():
        assert rows[cluster_id]["relaxation_proxy_score"] == pytest.approx(score)
        assert rows[cluster_id]["relaxation_proxy_stratum"] == label


def test_missingness_and_small_stratum_counts_are_explicit() -> None:
    result = strata()
    assert result["strata_counts"] == {
        "relaxation_proxy": {
            "relaxed_proxy": 4,
            "disturbed_proxy": 4,
            "relaxation_proxy_median_tie": 0,
        },
        "cool_core": {"CC": 3, "NCC": 5, "cool_core_missing": 0},
        "assembly_literature": {
            "no_class_assigned_in_frozen_source": 4,
            "sloshing_reported": 2,
            "sub_or_post_merger_reported": 2,
        },
        "published_stellar_profile": {"available": 5, "unavailable": 3},
        "boundary_background_method": {
            "standard_xmm_outer_background": 7,
            "rosat_exception": 1,
        },
    }
    assert result["missingness"]["assembly_negative_class_available"] is False
    assert result["missingness"]["assembly_unclassified_rows"] == 4
    assert result["missingness"]["missingness_imputed"] is False


def test_no_post_response_selection_rule_and_data_boundary_are_closed() -> None:
    frozen = config()
    rule = frozen["no_post_response_selection_rule"]
    assert rule["population_fixed_before_stratum_scoring"] is True
    assert rule["drop_or_replace_cluster_after_response_access"] is False
    assert rule["change_threshold_or_direction_after_response_access"] is False
    assert rule["merge_small_cells_after_response_access"] is False
    assert rule["impute_predictor_from_any_response"] is False
    assert rule["all_strata_results_must_be_reported_even_if_unfavorable"] is True
    assert rule["scientific_scoring_requires_a_separate_frozen_authorized_protocol"] is True
    assert frozen["data_boundary"] == {
        "development_cluster_identities_used": 8,
        "public_predictor_rows_used": 8,
        "target_or_response_rows_loaded": 0,
        "holdout_rows_loaded": 0,
        "confirmation_rows_loaded": 0,
        "independent_rows_loaded": 0,
        "target_scoring_calls": 0,
        "model_or_paid_calls": 0,
        "public_source_http_payloads_acquired": 2,
    }


def test_all_alternative_cause_preflights_are_ready_but_scoring_is_blocked() -> None:
    frozen = config()
    result = strata()
    matrix = preflight.build_cause_matrix(frozen, result)
    assert [row["cause_id"] for row in matrix["cause_rows"]] == preflight.CAUSES
    assert all(
        row["predictor_only_preflight_status"] == "EXECUTABLE_NOW" for row in matrix["cause_rows"]
    )
    assert all(
        row["scientific_comparison_status"].startswith("BLOCKED_") for row in matrix["cause_rows"]
    )
    assert matrix["summary"] == {
        "causes": 7,
        "predictor_only_preflights_executable_now": 7,
        "scientific_comparisons_blocked": 7,
        "scientific_comparisons_complete": 0,
        "ordinary_halo_comparison_role": (
            "already_frozen_elsewhere_in_CP4_not_reopened_or_rescored_here"
        ),
    }


def test_claim_ceiling_distinguishes_readiness_from_scientific_evidence() -> None:
    claims = config()["claim_boundary"]
    assert claims["CP5_11_predictor_definitions_frozen"] is True
    assert claims["CP5_11_predictor_labels_ready"] is True
    assert claims["CP5_11_scientific_stratum_scoring_complete"] is False
    assert claims["alternative_cause_planning_matrix_present"] is True
    assert claims["CP5_13_task_complete"] is False
    assert claims["CP5_13_scientific_comparisons_complete"] is False
    assert claims["cause_identified"] is False
    assert claims["candidate_supported_or_refuted"] is False
    assert claims["publication_readiness_changed"] is False
    assert claims["scientific_claim_allowed"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("title", "tampered title"),
        ("url", "https://example.invalid/tampered"),
        ("doi", "10.0000/tampered"),
        ("payload_bytes", 1),
        ("payload_sha256", "0" * 64),
        ("extracted_member_sha256", "0" * 64),
    ],
)
def test_public_source_nested_provenance_mutations_fail_closed(
    field: str, replacement: str | int
) -> None:
    bad = copy.deepcopy(raw_contract())
    bad["public_sources"][0][field] = replacement
    with pytest.raises(RuntimeError, match="authoritative public-source provenance changed"):
        preflight.validate_config_contract(bad)


def test_assembly_and_boundary_label_mutations_fail_closed() -> None:
    assembly = copy.deepcopy(raw_contract())
    assembly["cluster_metadata"][0]["assembly_literature_flag"] = "sloshing_reported"
    with pytest.raises(RuntimeError, match="assembly literature label changed"):
        preflight.validate_config_contract(assembly)

    boundary = copy.deepcopy(raw_contract())
    boundary["cluster_metadata"][0]["boundary_background_method"] = (
        "rosat_background_30pct_systematic"
    )
    with pytest.raises(RuntimeError, match="boundary background label changed"):
        preflight.validate_config_contract(boundary)


def test_missingness_policy_and_claim_ceiling_mutations_fail_closed() -> None:
    missingness = copy.deepcopy(raw_contract())
    missingness["missingness_policy"]["absent_public_predictor_remains_explicit_missing"] = False
    with pytest.raises(RuntimeError, match="missingness policy changed"):
        preflight.validate_config_contract(missingness)

    claims = copy.deepcopy(raw_contract())
    claims["claim_boundary"]["CP5_13_task_complete"] = True
    with pytest.raises(RuntimeError, match="claim boundary changed"):
        preflight.validate_config_contract(claims)


def test_cause_status_mutations_fail_closed() -> None:
    predictor_status = copy.deepcopy(raw_contract())
    predictor_status["alternative_cause_matrix"][0]["predictor_only_preflight_status"] = (
        "NOT_EXECUTABLE"
    )
    with pytest.raises(RuntimeError, match="alternative-cause status changed"):
        preflight.validate_config_contract(predictor_status)

    scientific_status = copy.deepcopy(raw_contract())
    scientific_status["alternative_cause_matrix"][0]["scientific_comparison_status"] = (
        "SCIENTIFICALLY_COMPLETE"
    )
    with pytest.raises(RuntimeError, match="alternative-cause status changed"):
        preflight.validate_config_contract(scientific_status)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["morphology"]["concentration_csb"].__setitem__("value", 0.131),
        lambda row: row.__setitem__("central_entropy_kev_cm2", 20.0),
        lambda row: row.__setitem__("published_stellar_profile_available", True),
        lambda row: row["aliases"].append("A1644_TAMPERED_ALIAS"),
        lambda row: row["provenance_ids"].append("TAMPERED_PROVENANCE"),
    ],
)
def test_every_cluster_nested_row_field_is_exactly_sealed(mutation: object) -> None:
    bad = copy.deepcopy(raw_contract())
    mutation(bad["cluster_metadata"][0])  # type: ignore[operator]
    with pytest.raises(RuntimeError, match="cluster metadata row changed"):
        preflight.validate_config_contract(bad)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.__setitem__("ordinary_explanation", "coherently rewritten cause"),
        lambda row: row["available_predictors"].append("invented_predictor"),
        lambda row: row["minimum_future_requirements"].append("invented_requirement"),
    ],
)
def test_every_cause_nested_row_field_is_exactly_sealed(mutation: object) -> None:
    bad = copy.deepcopy(raw_contract())
    mutation(bad["alternative_cause_matrix"][0])  # type: ignore[operator]
    with pytest.raises(RuntimeError, match="alternative-cause row changed"):
        preflight.validate_config_contract(bad)


def test_wrong_config_hash_and_implementation_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="config hash changed"):
        preflight.load_config(preflight.ROOT / preflight.CONFIG_PATH, "0" * 64)
    monkeypatch.setattr(preflight, "normalized_sha256", lambda _: "0" * 64)
    with pytest.raises(RuntimeError, match="implementation changed"):
        preflight.load_config(preflight.ROOT / preflight.CONFIG_PATH, CONFIG_SHA256)


def test_forbidden_response_derived_cluster_keys_fail_closed() -> None:
    frozen = config()
    bad = dict(frozen["cluster_metadata"][0])
    bad["target_residual"] = 1.0
    with pytest.raises(RuntimeError, match="cluster metadata keys changed"):
        preflight.validate_cluster_metadata(bad)


def test_atomic_writer_refuses_to_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    target = tmp_path / "receipt.json"
    preflight.write_json(target, {"version": 1})
    first = target.read_bytes()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        preflight.write_json(target, {"version": 2})
    assert target.read_bytes() == first
    assert json.loads(first) == {"version": 1}


def test_frozen_receipt_replays_exactly() -> None:
    result = preflight.check(
        preflight.ROOT / preflight.CONFIG_PATH,
        CONFIG_SHA256,
        preflight.ROOT / preflight.RECEIPT_PATH,
    )
    assert result == {
        "valid": True,
        "status": "predictor_preflight_pass_scientific_scoring_not_run",
        "decision": "CP5_11_STRATA_FROZEN_CP5_13_REMAINS_OPEN_PLANNING_MATRIX_ONLY",
        "CP5_11_predictor_definition_and_labels_ready": True,
        "CP5_11_scientific_stratum_scoring_complete": False,
        "alternative_cause_planning_matrix_present": True,
        "CP5_13_task_complete": False,
        "CP5_13_scientific_alternative_cause_comparison_complete": False,
        "target_or_response_rows_loaded": 0,
        "target_scoring_calls": 0,
        "scientific_claim_allowed": False,
    }
