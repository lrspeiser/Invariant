from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GravityCounterexamplePolicyError(ValueError):
    """Raised when a counterexample policy or evidence report is malformed."""


def load_counterexample_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_counterexample_policy(policy)
    return policy


def validate_counterexample_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != "sigma-gravity-empirical-counterexample-policy-1.0":
        raise GravityCounterexamplePolicyError("unsupported counterexample policy schema")
    if policy.get("status") != "frozen":
        raise GravityCounterexamplePolicyError("counterexample policy must be frozen")

    empirical = policy.get("empirical_evidence", {})
    required_false = (
        "single_counterexample_terminal_rejection_allowed",
        "single_counterexample_blocks_promotion_by_itself",
        "counterexample_count_alone_is_sufficient_for_rejection",
        "family_pruning_from_finite_empirical_sample_allowed",
        "post_response_exclusion_may_change_primary_result",
    )
    if any(empirical.get(name) is not False for name in required_false):
        raise GravityCounterexamplePolicyError(
            "empirical singleton, count-only, family-pruning, and post-response exclusions "
            "must remain non-terminal"
        )
    if empirical.get("single_counterexample_action") != "retain_and_audit":
        raise GravityCounterexamplePolicyError("one empirical counterexample must retain and audit")
    if empirical.get("minimum_uncertainty_resolved_counterexamples_for_pattern") != 2:
        raise GravityCounterexamplePolicyError(
            "the minimum empirical pattern size must remain two; count alone is still insufficient"
        )
    if empirical.get("unchanged_independent_replication_required_for_terminal_tested_scope_rejection") is not True:
        raise GravityCounterexamplePolicyError(
            "terminal tested-scope rejection requires unchanged independent replication"
        )
    if empirical.get("single_object_sensitive_formula_may_promote") is not False:
        raise GravityCounterexamplePolicyError(
            "single-object-sensitive formulas must remain unpromoted pending fresh testing"
        )

    theoretical = policy.get("hard_theoretical_veto", {})
    if theoretical.get("single_verified_witness_can_be_decisive") is not True:
        raise GravityCounterexamplePolicyError("verified theoretical witnesses must remain decisive")
    if theoretical.get("requires_declared_domain_membership") is not True:
        raise GravityCounterexamplePolicyError("theoretical vetoes require declared-domain membership")
    if theoretical.get("family_veto_requires_family_scope_proof") is not True:
        raise GravityCounterexamplePolicyError("family vetoes require a family-scope proof")
    if not theoretical.get("allowed_categories"):
        raise GravityCounterexamplePolicyError("hard theoretical veto categories are missing")

    required_fields = policy.get("required_empirical_report_fields")
    if not isinstance(required_fields, list) or len(required_fields) < 14:
        raise GravityCounterexamplePolicyError("required empirical reporting contract is incomplete")
    states = policy.get("allowed_empirical_states")
    if not isinstance(states, list) or len(states) < 8:
        raise GravityCounterexamplePolicyError("empirical evidence states are incomplete")


def _nonnegative_integer(report: dict[str, Any], name: str) -> int:
    value = report[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GravityCounterexamplePolicyError(f"{name} must be a nonnegative integer")
    return value


def _empirical_report_values(
    report: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    missing = [
        name for name in policy["required_empirical_report_fields"] if name not in report
    ]
    if missing:
        raise GravityCounterexamplePolicyError(
            "empirical report is missing fields: " + ", ".join(missing)
        )

    values = {
        name: _nonnegative_integer(report, name)
        for name in (
            "evaluable_objects",
            "raw_counterexample_count",
            "quality_verified_counterexample_count",
            "uncertainty_resolved_counterexample_count",
            "independent_failure_strata",
            "unchanged_independent_replication_failures",
        )
    }
    if values["evaluable_objects"] < 1:
        raise GravityCounterexamplePolicyError("an empirical report needs at least one object")
    if not (
        values["uncertainty_resolved_counterexample_count"]
        <= values["quality_verified_counterexample_count"]
        <= values["raw_counterexample_count"]
        <= values["evaluable_objects"]
    ):
        raise GravityCounterexamplePolicyError(
            "counterexample counts must be nested within raw count and evaluable objects"
        )

    improvement = report["aggregate_improvement_percent"]
    if isinstance(improvement, bool) or not isinstance(improvement, (int, float)):
        raise GravityCounterexamplePolicyError(
            "aggregate_improvement_percent must be numeric"
        )
    values["aggregate_improvement_percent"] = float(improvement)
    for name in (
        "quality_gate_passed",
        "strongest_baseline_failed",
        "leave_one_changes_sign",
        "trim_changes_sign",
        "object_level_records_preserved",
        "missing_quality_limited_records_preserved",
        "exclusions_frozen_before_response",
    ):
        if not isinstance(report[name], bool):
            raise GravityCounterexamplePolicyError(f"{name} must be boolean")
        values[name] = report[name]
    return values


def assess_counterexample_evidence(
    report: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    """Classify evidence without letting one empirical object terminate a formula.

    Empirical data can retire an unchanged tested representation only after an
    independent replication. Finite empirical evidence never automatically prunes
    the broader formula family. A verified in-domain theoretical contradiction is
    handled separately and can be decisive at candidate or proved family scope.
    """

    validate_counterexample_policy(policy)
    kind = report.get("evidence_kind")
    if kind == "theoretical":
        return _assess_theoretical(report, policy)
    if kind != "empirical":
        raise GravityCounterexamplePolicyError(
            "evidence_kind must be empirical or theoretical"
        )

    values = _empirical_report_values(report, policy)
    credible_count = values["uncertainty_resolved_counterexample_count"]
    sensitive = values["leave_one_changes_sign"] or values["trim_changes_sign"]
    complete = all(
        values[name]
        for name in (
            "object_level_records_preserved",
            "missing_quality_limited_records_preserved",
            "exclusions_frozen_before_response",
        )
    )
    negative_aggregate = (
        values["aggregate_improvement_percent"] < 0.0
        or values["strongest_baseline_failed"]
    )

    if not complete:
        status = "INCOMPLETE_EMPIRICAL_AUDIT_RETAINED"
        next_action = "complete the frozen object-level and data-quality audit"
    elif not values["quality_gate_passed"]:
        status = "QUALITY_LIMITED_EVIDENCE_RETAINED"
        next_action = "repeat unchanged on data that pass the frozen quality gate"
    elif credible_count == 0:
        status = "NO_EMPIRICAL_COUNTEREXAMPLE"
        next_action = "continue the predeclared promotion and independent-test gates"
    elif credible_count == 1:
        status = "ISOLATED_EMPIRICAL_COUNTEREXAMPLE_RETAINED"
        next_action = "audit the object and test the unchanged formula on fresh data"
    elif sensitive:
        status = "SINGLE_OBJECT_SENSITIVE_RETAINED"
        next_action = "withhold promotion and run an unchanged fresh-data test"
    elif not negative_aggregate:
        status = "SURVIVES_WITH_COUNTEREXAMPLES"
        next_action = "retain the formula and continue the predeclared promotion gates"
    elif values["unchanged_independent_replication_failures"] > 0:
        status = "REPLICATED_NEGATIVE_EVIDENCE_TESTED_REPRESENTATION"
        next_action = (
            "retire only the unchanged tested representation in its declared scope; "
            "preserve the family and failure record"
        )
    else:
        status = "ROBUST_SCOPED_NEGATIVE_EVIDENCE"
        next_action = (
            "preserve the formula and seek unchanged independent replication before "
            "terminal tested-scope rejection"
        )

    allowed_states = set(policy["allowed_empirical_states"])
    if status not in allowed_states:
        raise GravityCounterexamplePolicyError(f"unregistered empirical state: {status}")
    terminal_scope_rejection = (
        status == "REPLICATED_NEGATIVE_EVIDENCE_TESTED_REPRESENTATION"
    )
    promotion_eligible_from_counterexample_policy = status in {
        "NO_EMPIRICAL_COUNTEREXAMPLE",
        "ISOLATED_EMPIRICAL_COUNTEREXAMPLE_RETAINED",
        "SURVIVES_WITH_COUNTEREXAMPLES",
    }
    return {
        "schema_version": "sigma-gravity-counterexample-assessment-1.0",
        "status": status,
        "evidence_kind": "empirical",
        "raw_counterexample_count": values["raw_counterexample_count"],
        "quality_verified_counterexample_count": values[
            "quality_verified_counterexample_count"
        ],
        "uncertainty_resolved_counterexample_count": credible_count,
        "single_object_sensitive": sensitive,
        "terminal_rejection_in_tested_scope": terminal_scope_rejection,
        "candidate_pruned_globally": False,
        "formula_family_pruned": False,
        "promotion_eligible_from_counterexample_policy": (
            promotion_eligible_from_counterexample_policy
        ),
        "promotion_still_requires_all_other_frozen_gates": True,
        "post_response_exclusion_changes_primary_result": False,
        "next_action": next_action,
    }


def _assess_theoretical(
    report: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    required = ("violation_category", "verified", "within_declared_domain", "scope_proven")
    missing = [name for name in required if name not in report]
    if missing:
        raise GravityCounterexamplePolicyError(
            "theoretical report is missing fields: " + ", ".join(missing)
        )
    category = report["violation_category"]
    scope = report["scope_proven"]
    if category not in policy["hard_theoretical_veto"]["allowed_categories"]:
        raise GravityCounterexamplePolicyError("unregistered hard theoretical veto category")
    if not isinstance(report["verified"], bool) or not isinstance(
        report["within_declared_domain"], bool
    ):
        raise GravityCounterexamplePolicyError(
            "verified and within_declared_domain must be boolean"
        )
    if scope not in {"none", "candidate", "family"}:
        raise GravityCounterexamplePolicyError(
            "scope_proven must be none, candidate, or family"
        )

    decisive = report["verified"] and report["within_declared_domain"] and scope != "none"
    family_pruned = decisive and scope == "family"
    return {
        "schema_version": "sigma-gravity-counterexample-assessment-1.0",
        "status": "HARD_THEORETICAL_VETO" if decisive else "THEORETICAL_WARNING_RETAINED",
        "evidence_kind": "theoretical",
        "violation_category": category,
        "scope_proven": scope,
        "terminal_rejection_in_tested_scope": decisive,
        "candidate_pruned_globally": decisive,
        "formula_family_pruned": family_pruned,
        "next_action": (
            "record the verified veto at its proved scope"
            if decisive
            else "retain the candidate until the theoretical warning is verified in domain"
        ),
    }
