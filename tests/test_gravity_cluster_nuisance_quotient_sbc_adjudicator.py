from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_cluster_nuisance_quotient_sbc_adjudicator as adjudicator,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / adjudicator.CONFIG_PATH
RECEIPT = ROOT / adjudicator.RECEIPT_PATH


def load_contract() -> dict[str, object]:
    return adjudicator.load_config(CONFIG, adjudicator.file_sha256(CONFIG))


def test_contract_binds_all_ten_frozen_artifacts() -> None:
    config = load_contract()
    assert config["bound_sbc_artifacts"] == adjudicator.EXPECTED_SBC_ARTIFACTS
    assert len(config["bound_sbc_artifacts"]) == 10
    for binding in config["bound_sbc_artifacts"].values():
        path = ROOT / binding["path"]
        assert adjudicator.file_sha256(path) == binding["file_sha256"]


def test_strict_adjudication_recomputes_both_retained_failures() -> None:
    result = adjudicator.adjudicate(load_contract())
    assert result == {
        "v1": {
            "artifact_valid": True,
            "passed": False,
            "decision": "BOUNDED_SYNTHETIC_QUOTIENT_SBC_FAILED_RESULT_RETAINED",
            "synthetic_likelihood_evaluations": 884258,
        },
        "v2": {
            "artifact_valid": True,
            "passed": False,
            "decision": "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V2_FAILED_RESULT_RETAINED",
            "synthetic_likelihood_evaluations": 10735571,
        },
        "both_failed": True,
        "candidate_production_unlock": False,
        "newtonian_control_unlock": False,
    }


def test_sealed_receipts_exactly_reconstruct_from_npz_summaries() -> None:
    config = load_contract()
    v1_summary = adjudicator.validate_v1_result(
        ROOT / config["bound_sbc_artifacts"]["v1_result"]["path"], config
    )
    v2_summary = adjudicator.validate_v2_result(
        ROOT / config["bound_sbc_artifacts"]["v2_result"]["path"], config
    )
    actual_v1 = json.loads((ROOT / config["bound_sbc_artifacts"]["v1_receipt"]["path"]).read_text())
    actual_v2 = json.loads((ROOT / config["bound_sbc_artifacts"]["v2_receipt"]["path"]).read_text())
    assert actual_v1 == adjudicator.expected_v1_receipt(config, v1_summary)
    assert actual_v2 == adjudicator.expected_v2_receipt(config, v2_summary)


@pytest.mark.parametrize(
    "mutation",
    [
        "pass_status",
        "missing_evidence",
        "changed_counts",
        "changed_scenario",
        "changed_controls",
    ],
)
def test_v1_receipt_forgery_classes_are_rejected(mutation: str) -> None:
    config = load_contract()
    summary = adjudicator.validate_v1_result(
        ROOT / config["bound_sbc_artifacts"]["v1_result"]["path"], config
    )
    expected = adjudicator.expected_v1_receipt(config, summary)
    forged = copy.deepcopy(expected)
    if mutation == "pass_status":
        forged["status"] = "bounded_synthetic_sbc_passed_not_candidate_production"
    elif mutation == "missing_evidence":
        del forged["evidence"]["bounded_result"]
    elif mutation == "changed_counts":
        forged["counts"]["actual_total_synthetic_likelihood_evaluations"] -= 1
    elif mutation == "changed_scenario":
        forged["scenario_results"][0]["passed"] = True
    else:
        forged["controls"]["importance_reference_present"] = False
    forged["content_sha256"] = adjudicator.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(RuntimeError, match="does not exactly reconstruct"):
        adjudicator.require_exact_receipt(forged, expected, "forged V1 receipt")


@pytest.mark.parametrize(
    "mutation",
    [
        "pass_status",
        "missing_evidence",
        "changed_counts",
        "changed_scenario",
        "changed_controls",
    ],
)
def test_v2_receipt_forgery_classes_are_rejected(mutation: str) -> None:
    config = load_contract()
    summary = adjudicator.validate_v2_result(
        ROOT / config["bound_sbc_artifacts"]["v2_result"]["path"], config
    )
    expected = adjudicator.expected_v2_receipt(config, summary)
    forged = copy.deepcopy(expected)
    if mutation == "pass_status":
        forged["status"] = "bounded_synthetic_sbc_v2_passed_not_candidate_production"
    elif mutation == "missing_evidence":
        del forged["evidence"]["bounded_v2_result"]
    elif mutation == "changed_counts":
        forged["counts"]["actual_total_synthetic_likelihood_evaluations"] -= 1
    elif mutation == "changed_scenario":
        forged["scenario_results"][0]["passed"] = True
    else:
        forged["controls"]["reference_uses_candidate_transition"] = True
    forged["content_sha256"] = adjudicator.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(RuntimeError, match="does not exactly reconstruct"):
        adjudicator.require_exact_receipt(forged, expected, "forged V2 receipt")


def test_npz_v1_v2_cross_swaps_are_rejected() -> None:
    config = load_contract()
    v1_result = ROOT / config["bound_sbc_artifacts"]["v1_result"]["path"]
    v2_result = ROOT / config["bound_sbc_artifacts"]["v2_result"]["path"]
    with pytest.raises(RuntimeError, match="NPZ keys changed"):
        adjudicator.validate_v1_result(v2_result, config)
    with pytest.raises(RuntimeError, match="NPZ keys changed"):
        adjudicator.validate_v2_result(v1_result, config)


def test_machine_receipt_is_exact_and_locks_both_production_paths() -> None:
    checked = adjudicator.check_receipt(CONFIG, adjudicator.file_sha256(CONFIG), RECEIPT)
    assert checked["valid"] is True
    assert checked["passed"] is False
    assert checked["v1_passed"] is False
    assert checked["v2_passed"] is False
    assert checked["candidate_production_unlock"] is False
    assert checked["newtonian_control_unlock"] is False
    assert checked["real_rows_loaded"] == 0
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["decision"] == adjudicator.DECISION
    assert receipt["claim_boundary"] == adjudicator.CLAIM_BOUNDARY
    assert receipt["data_boundary"] == adjudicator.DATA_BOUNDARY


def test_new_files_contain_no_work_or_real_target_bindings() -> None:
    for path in (CONFIG, RECEIPT, Path(adjudicator.__file__)):
        text = path.read_text(encoding="utf-8").replace("\\", "/").lower()
        assert "work/" not in text
    assert adjudicator.DATA_BOUNDARY["real_development_rows_loaded"] == 0
    assert adjudicator.DATA_BOUNDARY["real_holdout_rows_loaded"] == 0
    assert adjudicator.DATA_BOUNDARY["real_confirmation_rows_loaded"] == 0
    assert adjudicator.DATA_BOUNDARY["real_independent_rows_loaded"] == 0
