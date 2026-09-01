from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler import open_gravity_audit_recovery_program_v1 as recovery


def test_exact_279_concept_recovery_partition() -> None:
    receipt = recovery.build_receipt()
    coverage = receipt["coverage"]
    assert coverage["concepts"] == 279
    assert coverage["class_counts"] == recovery._EXPECTED_CLASS_COUNTS
    assignments = coverage["assignments"]
    assert len({row["concept_id"] for row in assignments}) == 279
    assert all(
        row["empirical_result"] == "NO_EMPIRICAL_LOSS_IN_BOUND_CAMPAIGN" for row in assignments
    )


def test_classifier_covers_exact_boundary_cases() -> None:
    assert recovery.classify_concept("TW2-A01-D08") == "RC01_TWELL_STATIC_MISSING_DRIVERS"
    assert recovery.classify_concept("TW2-A19-D20") == "RC01_TWELL_STATIC_MISSING_DRIVERS"
    assert recovery.classify_concept("TW2-A15-D01") == "RC02_TWELL_DYNAMIC_HISTORY"
    assert recovery.classify_concept("TW2-A18-D20") == "RC02_TWELL_DYNAMIC_HISTORY"
    assert recovery.classify_concept("X20") == "RC03_TWELL_COMPOUNDS"
    assert recovery.classify_concept("GP01-T1") == "RC04_GP01_TRANSPORT"
    assert recovery.classify_concept("GP01-AQUAL") == "RC05_GP01_AQUAL_CONTROL"
    assert recovery.classify_concept("GP01-ACTION-PLACEHOLDER") == "RC06_GP01_ACTION_REPAIR"
    with pytest.raises(recovery.AuditRecoveryError):
        recovery.classify_concept("TW2-A01-D01")


def test_historical_families_all_remain_active() -> None:
    config = recovery.load_config()
    rows = config["historical_recovery_workstreams"]
    assert len(rows) == 12
    assert all(row["status"].startswith("ACTIVE") for row in rows)
    assert all(
        row["empirical_next"] and row["theory_next"] and row["publication_hook"] for row in rows
    )


def test_dual_grades_cannot_be_collapsed() -> None:
    config = recovery.load_config()
    policy = config["dual_grade_policy"]
    assert "DATA_REJECTED_EXACT_IMPLEMENTATION" in policy["empirical_grades"]
    assert "THEORY_OBSTRUCTION_EXACT_BRANCH" in policy["theory_grades"]
    assert policy["rule"].startswith("A theory grade never overwrites")
    mutated = copy.deepcopy(config)
    mutated["dual_grade_policy"]["rule"] = "Theory failure erases the empirical result."
    with pytest.raises(recovery.AuditRecoveryError):
        recovery.validate_config(mutated)


def test_publication_lead_retains_gp01_counterexample() -> None:
    leads = {row["lead_id"]: row for row in recovery.build_receipt()["publication_leads"]}
    lead = leads["PL01_CLUSTER_DYNAMICAL_STATE_RESIDUAL"]
    assert lead["state"] == "DEVELOPMENT_ASSOCIATION"
    assert lead["evidence"]["objects"] == 8
    assert lead["evidence"]["entropy_loss_difference_exact_p"] == pytest.approx(0.00873015873015873)
    assert lead["evidence"]["elliptic_to_equilibrium_robust_loss_ratio"] > 1.0
    assert "0 of 8" in lead["counterexample"]


def test_no_scientific_access_or_scoring() -> None:
    receipt = recovery.build_receipt()
    assert set(receipt["access_accounting"].values()) == {0}
    assert receipt["claim_boundary"]["does_not_establish"]


def test_receipt_round_trip(tmp_path, monkeypatch) -> None:
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(recovery, "OUTPUT_PATH", output)
    assert recovery.write_receipt() == "CREATED"
    recovery.validate_receipt()
    assert recovery.write_receipt() == "EXISTING_IDENTICAL"
