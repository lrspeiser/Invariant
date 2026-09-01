from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_split_gate_range_source_novelty_benchmark as novelty,
)


def test_config_identity_and_primary_source_gate() -> None:
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


def test_primary_paper_inventory_is_exact_and_cautious() -> None:
    papers = novelty.load_config()["primary_literature"]
    assert len(papers) == 8
    assert {item["arxiv"] for item in papers} == {
        "astro-ph/0309300",
        "1001.4525",
        "0905.2943",
        "1306.6401",
        "1611.08279",
        "2305.07725",
        "2603.13986v2",
        "2604.20292",
    }
    assert all(item["exact_theorem_found"] is False for item in papers)


def test_predecessor_is_exact_committed_packet() -> None:
    receipt = novelty.build_receipt()
    binding = receipt["predecessor_binding"]
    assert binding["commit"] == "ab54cf4a1eadec793df8cba61ee0fb70002bfb0e"
    assert binding["config_sha256"] == (
        "66cdf70b39a6402cf1da38ee0df1f88d1b95a58f2649d881813603554facb09f"
    )
    assert binding["receipt_content_sha256"] == (
        "50ec6d65d3dc09b993c19e57638ace6b6585054ba1e80d8300b93833d4402b2d"
    )


def test_symbolic_benchmarks_all_pass() -> None:
    checks = novelty._symbolic_checks()
    assert len(checks) == 9
    assert all(checks.values())


def test_independent_numeric_slopes_match_exact_exponents() -> None:
    evidence = novelty._numeric_evidence(novelty.load_config())
    assert evidence["all_passed"] is True
    assert len(evidence["slope_records"]) == 4
    for record in evidence["slope_records"]:
        assert record["max_abs_error"] < 1.0e-11
        assert record["passed"] is True


def test_threshold_controls_retain_all_three_behaviors() -> None:
    records = novelty._numeric_evidence(novelty.load_config())["threshold_records"]
    assert len(records) == 12
    assert {item["behavior"] for item in records} == {
        "DECREASING_SUBCRITICAL",
        "COEFFICIENT_DEPENDENT_CRITICAL",
        "INCREASING_SUPERCRITICAL",
    }
    assert all(item["passed"] for item in records)


def test_product_is_scoped_to_architecture_not_universal_physics() -> None:
    claims = novelty.load_config()["claim_boundary"]
    assert claims["architecture_class_universality_only"] is True
    assert claims["universal_physics_claim"] is False
    assert claims["full_coupled_range_derived"] is False
    assert claims["full_action_health"] is False


def test_novelty_and_publication_are_not_overclaimed() -> None:
    config = novelty.load_config()
    claims = config["claim_boundary"]
    assert claims["candidate_corollary_not_found_in_reviewed_set"] is True
    assert claims["historical_novelty_established"] is False
    assert claims["independent_expert_review_passed"] is False
    assert claims["publication_ready"] is False
    assert config["search_protocol"]["limitation"].startswith("A targeted search")


def test_zero_observational_access() -> None:
    receipt = novelty.build_receipt()
    assert receipt["access_ledger"] == {
        "observational_files_opened": 0,
        "observational_rows_read": 0,
        "scores_computed": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }


def test_receipt_is_deterministic_and_self_hashed() -> None:
    first = novelty.build_receipt()
    second = novelty.build_receipt()
    assert first == second
    assert first["content_sha256"] == novelty._self_hash(first)
    assert first["checks_passed"] == 16


def test_coherent_claim_forgery_does_not_equal_rebuild() -> None:
    receipt = novelty.build_receipt()
    forged = copy.deepcopy(receipt)
    forged["claim_boundary"]["historical_novelty_established"] = True
    forged["content_sha256"] = novelty._self_hash(forged)
    assert forged["content_sha256"] == novelty._self_hash(forged)
    assert forged != novelty.build_receipt()


def test_local_integrity_pins_config_module_and_tests() -> None:
    root = novelty._repo_root()
    binding = novelty._validate_local_integrity(root)
    assert binding["config_raw_sha256"] == novelty.EXPECTED_CONFIG_RAW_SHA256
    assert binding["module_semantic_sha256"] == novelty.EXPECTED_MODULE_SEMANTIC_SHA256
    assert binding["test_raw_sha256"] == novelty.EXPECTED_TEST_RAW_SHA256


def test_config_byte_mutation_rejects_before_semantic_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = novelty._repo_root()
    target = tmp_path / novelty.CONFIG_PATH
    target.parent.mkdir(parents=True)
    config = json.loads((root / novelty.CONFIG_PATH).read_text(encoding="utf-8"))
    config["claim_boundary"]["publication_ready"] = True
    target.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(novelty, "EXPECTED_CONFIG_RAW_SHA256", "0" * 64)
    with pytest.raises(novelty.SplitGateRangeSourceNoveltyError, match="config semantics changed"):
        novelty.load_config(tmp_path)


def test_write_replay_and_check_are_no_clobber() -> None:
    assert novelty.write_receipt() == "EXISTING_IDENTICAL"
    stored = novelty.validate_receipt()
    assert stored == novelty.build_receipt()


def test_decision_is_theory_review_not_model_success() -> None:
    receipt = novelty.build_receipt()
    assert receipt["decision"] == novelty.DECISION
    assert receipt["adjudication"]["publication_value"] == (
        "WORTH_INDEPENDENT_EXPERT_REVIEW_AND_POSSIBLE_NARROW_THEORY_NOTE"
    )
    assert receipt["adjudication"]["physical_theory_status"] == "NOT_ESTABLISHED"
