import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_counterexample_policy import (
    GravityCounterexamplePolicyError,
    assess_counterexample_evidence,
    load_counterexample_policy,
    validate_counterexample_policy,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "gravity_empirical_counterexample_policy_v1.json"


def _empirical_report(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "evidence_kind": "empirical",
        "evaluable_objects": 100,
        "raw_counterexample_count": 1,
        "quality_verified_counterexample_count": 1,
        "uncertainty_resolved_counterexample_count": 1,
        "aggregate_improvement_percent": 12.0,
        "quality_gate_passed": True,
        "strongest_baseline_failed": False,
        "leave_one_changes_sign": False,
        "trim_changes_sign": False,
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    report.update(overrides)
    return report


def test_frozen_policy_forbids_single_empirical_counterexample_rejection() -> None:
    policy = load_counterexample_policy(POLICY_PATH)
    empirical = policy["empirical_evidence"]
    assert empirical["single_counterexample_action"] == "retain_and_audit"
    assert empirical["single_counterexample_terminal_rejection_allowed"] is False
    assert empirical["single_counterexample_blocks_promotion_by_itself"] is False
    assert empirical["counterexample_count_alone_is_sufficient_for_rejection"] is False
    assert empirical["family_pruning_from_finite_empirical_sample_allowed"] is False


def test_policy_rejects_tamper_that_turns_a_singleton_into_a_kill_rule() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    corrupted = deepcopy(policy)
    corrupted["empirical_evidence"][
        "single_counterexample_terminal_rejection_allowed"
    ] = True
    with pytest.raises(GravityCounterexamplePolicyError):
        validate_counterexample_policy(corrupted)


def test_one_empirical_counterexample_is_retained_and_does_not_block_promotion() -> None:
    result = assess_counterexample_evidence(
        _empirical_report(), load_counterexample_policy(POLICY_PATH)
    )
    assert result["status"] == "ISOLATED_EMPIRICAL_COUNTEREXAMPLE_RETAINED"
    assert result["terminal_rejection_in_tested_scope"] is False
    assert result["candidate_pruned_globally"] is False
    assert result["formula_family_pruned"] is False
    assert result["promotion_eligible_from_counterexample_policy"] is True


def test_one_counterexample_cannot_kill_even_when_aggregate_score_is_negative() -> None:
    result = assess_counterexample_evidence(
        _empirical_report(
            evaluable_objects=1,
            aggregate_improvement_percent=-80.0,
            strongest_baseline_failed=True,
        ),
        load_counterexample_policy(POLICY_PATH),
    )
    assert result["status"] == "ISOLATED_EMPIRICAL_COUNTEREXAMPLE_RETAINED"
    assert result["terminal_rejection_in_tested_scope"] is False


def test_single_object_sensitive_result_is_retained_without_promotion() -> None:
    result = assess_counterexample_evidence(
        _empirical_report(
            raw_counterexample_count=25,
            quality_verified_counterexample_count=20,
            uncertainty_resolved_counterexample_count=12,
            leave_one_changes_sign=True,
        ),
        load_counterexample_policy(POLICY_PATH),
    )
    assert result["status"] == "SINGLE_OBJECT_SENSITIVE_RETAINED"
    assert result["terminal_rejection_in_tested_scope"] is False
    assert result["promotion_eligible_from_counterexample_policy"] is False


def test_many_counterexamples_can_still_survive_on_aggregate() -> None:
    result = assess_counterexample_evidence(
        _empirical_report(
            raw_counterexample_count=30,
            quality_verified_counterexample_count=28,
            uncertainty_resolved_counterexample_count=12,
            aggregate_improvement_percent=8.0,
        ),
        load_counterexample_policy(POLICY_PATH),
    )
    assert result["status"] == "SURVIVES_WITH_COUNTEREXAMPLES"
    assert result["promotion_eligible_from_counterexample_policy"] is True


def test_dataset_negative_is_scoped_until_unchanged_independent_replication() -> None:
    policy = load_counterexample_policy(POLICY_PATH)
    scoped = assess_counterexample_evidence(
        _empirical_report(
            raw_counterexample_count=80,
            quality_verified_counterexample_count=75,
            uncertainty_resolved_counterexample_count=50,
            aggregate_improvement_percent=-20.0,
            strongest_baseline_failed=True,
            independent_failure_strata=4,
        ),
        policy,
    )
    assert scoped["status"] == "ROBUST_SCOPED_NEGATIVE_EVIDENCE"
    assert scoped["terminal_rejection_in_tested_scope"] is False
    assert scoped["formula_family_pruned"] is False

    replicated = assess_counterexample_evidence(
        _empirical_report(
            raw_counterexample_count=80,
            quality_verified_counterexample_count=75,
            uncertainty_resolved_counterexample_count=50,
            aggregate_improvement_percent=-20.0,
            strongest_baseline_failed=True,
            independent_failure_strata=4,
            unchanged_independent_replication_failures=1,
        ),
        policy,
    )
    assert replicated["status"] == "REPLICATED_NEGATIVE_EVIDENCE_TESTED_REPRESENTATION"
    assert replicated["terminal_rejection_in_tested_scope"] is True
    assert replicated["candidate_pruned_globally"] is False
    assert replicated["formula_family_pruned"] is False


def test_failed_quality_or_incomplete_audit_always_retains_formula() -> None:
    policy = load_counterexample_policy(POLICY_PATH)
    quality_limited = assess_counterexample_evidence(
        _empirical_report(
            raw_counterexample_count=90,
            quality_verified_counterexample_count=80,
            uncertainty_resolved_counterexample_count=70,
            aggregate_improvement_percent=-50.0,
            strongest_baseline_failed=True,
            quality_gate_passed=False,
        ),
        policy,
    )
    assert quality_limited["status"] == "QUALITY_LIMITED_EVIDENCE_RETAINED"
    assert quality_limited["terminal_rejection_in_tested_scope"] is False

    incomplete = assess_counterexample_evidence(
        _empirical_report(object_level_records_preserved=False), policy
    )
    assert incomplete["status"] == "INCOMPLETE_EMPIRICAL_AUDIT_RETAINED"
    assert incomplete["terminal_rejection_in_tested_scope"] is False


def test_verified_theoretical_veto_is_separate_and_scope_bound() -> None:
    policy = load_counterexample_policy(POLICY_PATH)
    candidate = assess_counterexample_evidence(
        {
            "evidence_kind": "theoretical",
            "violation_category": "ghost_or_unbounded_hamiltonian",
            "verified": True,
            "within_declared_domain": True,
            "scope_proven": "candidate",
        },
        policy,
    )
    assert candidate["status"] == "HARD_THEORETICAL_VETO"
    assert candidate["candidate_pruned_globally"] is True
    assert candidate["formula_family_pruned"] is False

    family = assess_counterexample_evidence(
        {
            "evidence_kind": "theoretical",
            "violation_category": "broken_conservation_identity",
            "verified": True,
            "within_declared_domain": True,
            "scope_proven": "family",
        },
        policy,
    )
    assert family["formula_family_pruned"] is True
