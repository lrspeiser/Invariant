from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_deep_aqual_transition_novelty_benchmark as novelty,
)


def test_config_identity_and_admission_policy() -> None:
    config = novelty.load_config()
    assert config["artifact_id"] == novelty.ARTIFACT_ID
    assert config["admission_policy"]["source_class"] == (
        "PRIMARY_PAPERS_PLUS_EXACT_ANALYTIC_AND_NUMERIC_BENCHMARKS"
    )
    assert (
        config["admission_policy"]["observational_data_required_for_this_theory_only_step"] is False
    )
    assert (
        config["admission_policy"][
            "future_observational_promotion_requires_real_source_and_response_data"
        ]
        is True
    )


def test_primary_source_inventory_and_direct_overlap_are_exact() -> None:
    papers = novelty.load_config()["primary_literature"]
    assert len(papers) == 7
    assert {item["id"] for item in papers} == {
        "DOI:10.1086/162570",
        "astro-ph/0403694",
        "astro-ph/0512425",
        "gr-qc/0607055",
        "0705.4043",
        "0708.0561",
        "hep-th/9904176",
    }
    direct = [item for item in papers if item["overlap"] == "DIRECT_MATHEMATICAL_SUBSTANCE"]
    assert [item["id"] for item in direct] == ["0705.4043"]
    assert all(item["exact_wording_found"] is False for item in papers)


def test_committed_predecessor_binding_is_exact() -> None:
    receipt = novelty.build_receipt()
    binding = receipt["predecessor_binding"]
    assert binding["commit"] == "7b330ab45c41f863c2e84f2c39f01fc0519f013d"
    assert binding["config_sha256"] == (
        "2d5a67b6231c5fafabffdef5b369e9a42c73e179fd943c2c2925373a0c270277"
    )
    assert binding["receipt_content_sha256"] == (
        "2a8007fc24441a78bd2b95638798b557a551a501c271a6ba0c09869fc0bfb6a7"
    )


def test_symbolic_benchmarks_rederive_the_transition_obstruction() -> None:
    checks = novelty._symbolic_checks()
    assert len(checks) == 8
    assert all(checks.values())


def test_numeric_power_probes_cover_sublinear_linear_and_superlinear() -> None:
    evidence = novelty._numeric_power_evidence(novelty.load_config())
    assert evidence["all_passed"] is True
    assert len(evidence["records"]) == 5
    assert {record["derivative_behavior"] for record in evidence["records"]} == {
        "DIVERGES_TOWARD_ZERO",
        "FINITE_CONSTANT",
        "VANISHES_TOWARD_ZERO",
    }
    for record in evidence["records"]:
        assert record["C_log_slope"] == pytest.approx(record["p"], abs=1e-12)
        assert record["determinant_log_slope"] == pytest.approx(4.0 * record["p"], abs=1e-12)


def test_regulator_repairs_transition_but_retains_costs() -> None:
    evidence = novelty._numeric_regulator_evidence(novelty.load_config())
    assert evidence["all_passed"] is True
    assert evidence["transition_passed"] is True
    assert evidence["accuracy_cost_passed"] is True
    assert evidence["timelike_cone_cost_passed"] is True
    assert evidence["spacelike_records"][0]["C"] == pytest.approx(0.2)
    assert evidence["spacelike_records"][0]["H"] == pytest.approx(0.2)
    assert all(
        item["speed_squared"] > 1.0 for item in evidence["timelike_records"] if item["X"] > 0
    )


def test_static_aqual_is_retained_not_eliminated() -> None:
    config = novelty.load_config()
    assert config["claim_boundary"]["static_aqual_invalidated"] is False
    assert config["adjudication"]["static_aqual_observational_testing_disposition"].startswith(
        "RETAIN_STATIC_AND_3D_AQUAL_TESTS"
    )
    assert (
        "rather than rejecting the entire family" in config["adjudication"]["next_required_work"][1]
    )


def test_novelty_verdict_withholds_standalone_publication() -> None:
    config = novelty.load_config()
    claims = config["claim_boundary"]
    assert claims["mathematical_substance_preexisting"] is True
    assert claims["exact_verbatim_duplicate_found"] is False
    assert claims["historical_novelty_established"] is False
    assert claims["standalone_publication_candidate"] is False
    assert claims["useful_internal_design_constraint"] is True
    assert config["adjudication"]["publication_value"] == (
        "DO_NOT_PROMOTE_AS_STANDALONE_NOTE_USE_AS_CITED_DESIGN_CONSTRAINT"
    )


def test_theorem_does_not_overclaim_action_or_observation() -> None:
    claims = novelty.load_config()["claim_boundary"]
    assert claims["regulator_phenomenologically_derived"] is False
    assert claims["full_action_health"] is False
    assert claims["observational_support"] is False
    assert claims["modified_gravity_success"] is False
    assert claims["publication_ready"] is False


def test_receipt_is_deterministic_and_self_hashed() -> None:
    first = novelty.build_receipt()
    second = novelty.build_receipt()
    assert first == second
    assert first["content_sha256"] == novelty._self_hash(first)
    assert first["checks_passed"] == 16


def test_coherent_claim_forgery_does_not_equal_rebuild() -> None:
    forged = copy.deepcopy(novelty.build_receipt())
    forged["claim_boundary"]["standalone_publication_candidate"] = True
    forged["claim_boundary"]["historical_novelty_established"] = True
    forged["content_sha256"] = novelty._self_hash(forged)
    assert forged["content_sha256"] == novelty._self_hash(forged)
    assert forged != novelty.build_receipt()


def test_local_integrity_pins_config_module_and_tests() -> None:
    binding = novelty._validate_local_integrity()
    assert binding["config_raw_sha256"] == novelty.EXPECTED_CONFIG_RAW_SHA256
    assert binding["module_semantic_sha256"] == novelty.EXPECTED_MODULE_SEMANTIC_SHA256
    assert binding["test_raw_sha256"] == novelty.EXPECTED_TEST_RAW_SHA256


def test_zero_observational_access() -> None:
    assert novelty.build_receipt()["access_ledger"] == {
        "observational_files_opened": 0,
        "observational_rows_read": 0,
        "scores_computed": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }


def test_write_replay_and_check_are_no_clobber() -> None:
    assert novelty.write_receipt() == "EXISTING_IDENTICAL"
    assert novelty.validate_receipt() == novelty.build_receipt()
