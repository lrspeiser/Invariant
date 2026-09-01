from __future__ import annotations

import copy
import math

import pytest

from sigma_theory_compiler import open_gravity_rg_things_five_object_2d_phenotype_v1 as phenotype


def test_config_freezes_source_paper_benchmark_admission() -> None:
    config = phenotype.load_config(verify_package=False)
    admission = config["admission_rule"]
    assert len(admission["primary_measurement_papers"]) == 3
    assert len(admission["independent_known_answer_benchmarks"]) == 3
    assert admission["dimensional_claim"] == (
        "MODEL_LIFTED_2P5D_SOURCE_PLUS_PROJECTED_2D_VELOCITY_FIELD"
    )
    assert admission["general_3d_validation"] is False


def test_receipt_rebuilds_five_exact_objects() -> None:
    receipt = phenotype.build_receipt()
    rows = receipt["object_feature_rows"]
    assert [row["object_id"] for row in rows] == [
        "NGC2903",
        "NGC2976",
        "NGC3198",
        "NGC3521",
        "NGC4214",
    ]
    assert all(len(row["features"]) == 11 for row in rows)
    assert all(len(row["responses"]) == 2 for row in rows)


def test_outcome_partition_retains_every_counterexample() -> None:
    receipt = phenotype.build_receipt()
    by_id = {row["object_id"]: row["outcome_partition"] for row in receipt["object_feature_rows"]}
    assert by_id == {
        "NGC2903": "BEATS_NEWTON_ONLY",
        "NGC2976": "BEATS_BOTH",
        "NGC3198": "BEATS_NEWTON_ONLY",
        "NGC3521": "BEATS_BOTH",
        "NGC4214": "BEATS_RAR_ONLY",
    }
    assert receipt["outcome_counts"] == {
        "BEATS_BOTH": 2,
        "BEATS_NEWTON_ONLY": 2,
        "BEATS_RAR_ONLY": 1,
        "BEATS_NEITHER": 0,
    }


def test_density_imprint_is_exact_but_not_familywise_discovery() -> None:
    result = phenotype.build_receipt()["mechanism_density_imprint"]
    assert result["feature_id"] == "log10_rho_midplane_msun_pc3"
    assert result["response_axis"] == "rg_fractional_improvement_over_newton"
    assert result["exact_monotonic_rank_relation"] is True
    assert math.isclose(result["spearman_rho"], -1.0, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(result["exact_two_sided_p"], 1.0 / 60.0)
    assert math.isclose(result["bonferroni_p"], 22.0 / 60.0)
    assert result["familywise_significant"] is False
    assert result["post_response"] is True


def test_no_scalar_source_feature_explains_rar_variation() -> None:
    receipt = phenotype.build_receipt()
    assert receipt["familywise_significant_associations"] == []
    conclusions = receipt["diagnostic_conclusions"]
    assert conclusions["source_feature_explains_rg_vs_rar_variation"] is False
    assert conclusions["publishable_subclass_identified"] is False
    assert conclusions["density_imprint_is_independent_rg_evidence"] is False


def test_resolved_counts_are_preserved() -> None:
    conclusions = phenotype.build_receipt()["diagnostic_conclusions"]
    assert conclusions["rg_beats_newton_objects"] == [
        "NGC2903",
        "NGC2976",
        "NGC3198",
        "NGC3521",
    ]
    assert conclusions["rg_beats_rar_objects"] == ["NGC2976", "NGC3521", "NGC4214"]
    assert conclusions["rg_beats_both_objects"] == ["NGC2976", "NGC3521"]


def test_next_builder_is_source_and_paper_gated() -> None:
    next_test = phenotype.build_receipt()["next_test_contract"]
    assert next_test["source_before_builder"] is True
    assert next_test["required_primary_papers"] is True
    assert next_test["required_independent_benchmarks"] is True
    assert next_test["minimum_new_response_blind_objects"] == 5
    assert next_test["general_3d_claim_allowed"] is False


def test_zero_new_access_and_no_retuning() -> None:
    config = phenotype.load_config(verify_package=False)
    assert set(config["access_scope"].values()) == {0}
    statistics = config["statistical_contract"]
    assert statistics["new_formula_fit"] is False
    assert statistics["classifier_fit"] is False
    assert statistics["threshold_tuning"] is False
    assert statistics["object_pruning"] is False


def test_config_mutation_fails_closed() -> None:
    config = phenotype.load_config(verify_package=False)
    forged = copy.deepcopy(config)
    forged["claim_boundary"]["publication_ready"] = True
    with pytest.raises(phenotype.PhenotypeDiagnosticError):
        phenotype.validate_config(forged)


def test_receipt_mutation_fails_closed() -> None:
    receipt = phenotype.build_receipt()
    forged = copy.deepcopy(receipt)
    forged["diagnostic_conclusions"]["publishable_subclass_identified"] = True
    forged["content_sha256"] = phenotype.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(phenotype.PhenotypeDiagnosticError):
        phenotype.validate_receipt(forged)


def test_deterministic_rebuild_and_self_hash() -> None:
    first = phenotype.build_receipt()
    second = phenotype.build_receipt()
    assert first == second
    assert first["content_sha256"] == phenotype.content_sha256(
        {key: value for key, value in first.items() if key != "content_sha256"}
    )
    phenotype.validate_receipt(first)
