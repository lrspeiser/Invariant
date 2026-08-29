from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    gravity_cluster_nuisance_quotient_sbc_v3_adjudicator as adjudicator,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / adjudicator.CONFIG_PATH
RECEIPT = ROOT / adjudicator.RECEIPT_PATH


def load_contract() -> dict[str, object]:
    return adjudicator.load_config(CONFIG, adjudicator.file_sha256(CONFIG))


def validated_summary(config: dict[str, object]) -> dict[str, object]:
    binding = config["v3_artifacts"]["v3_result"]
    return adjudicator.validate_v3_result(ROOT / binding["path"], config)


def test_contract_binds_v3_predecessor_and_all_predecessor_dependencies() -> None:
    config = load_contract()
    assert config["v3_artifacts"] == adjudicator.EXPECTED_V3_ARTIFACTS
    assert config["predecessor_adjudicator"] == (adjudicator.EXPECTED_PREDECESSOR_ADJUDICATOR)
    assert config["predecessor_sbc_artifacts"] == (adjudicator.predecessor.EXPECTED_SBC_ARTIFACTS)
    assert len(config["v3_artifacts"]) == 5
    assert len(config["predecessor_adjudicator"]) == 4
    assert len(config["predecessor_sbc_artifacts"]) == 10
    for group in ("v3_artifacts", "predecessor_adjudicator", "predecessor_sbc_artifacts"):
        for binding in config[group].values():
            assert adjudicator.file_sha256(ROOT / binding["path"]) == binding["file_sha256"]


def test_v3_npz_strictly_reconstructs_synthetic_pass() -> None:
    config = load_contract()
    summary = validated_summary(config)
    assert summary["passed"] is True
    assert summary["decision"] == (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V3_PASSED_NOT_PHYSICS_OR_PRODUCTION"
    )
    assert all(row["passed"] for row in summary["scenario_results"])
    assert summary["call_accounting"]["actual_total_synthetic_likelihood_evaluations"] == 24_896_774


def test_truth_seed_shapes_and_v2_reference_are_exact() -> None:
    config = load_contract()
    v3_path = ROOT / config["v3_artifacts"]["v3_result"]["path"]
    v2_path = ROOT / config["predecessor_sbc_artifacts"]["v2_result"]["path"]
    with (
        np.load(v3_path, allow_pickle=False) as v3_archive,
        np.load(v2_path, allow_pickle=False) as v2_archive,
    ):
        truth, scenarios = adjudicator.expected_truth_units()
        assert np.array_equal(v3_archive["truth_units"], truth)
        assert np.array_equal(v3_archive["scenario_indices"], scenarios)
        for name in (
            "truth_units",
            "scenario_indices",
            "reference_normalized_ranks",
            "reference_coverage",
            "reference_tie_mass",
        ):
            assert np.array_equal(v3_archive[name], v2_archive[name])


def test_v3_receipt_exactly_reconstructs_from_sealed_result() -> None:
    config = load_contract()
    summary = validated_summary(config)
    actual = json.loads((ROOT / config["v3_artifacts"]["v3_receipt"]["path"]).read_text())
    assert actual == adjudicator.expected_v3_receipt(config, summary)


@pytest.mark.parametrize(
    "mutation",
    [
        "forged_pass",
        "missing_result_evidence",
        "missing_all_evidence",
        "missing_scenarios",
        "changed_counts",
        "changed_controls",
        "changed_scenario",
    ],
)
def test_v3_receipt_forgery_classes_are_rejected(mutation: str) -> None:
    config = load_contract()
    expected = adjudicator.expected_v3_receipt(config, validated_summary(config))
    forged = copy.deepcopy(expected)
    if mutation == "forged_pass":
        forged["status"] = "bounded_synthetic_sbc_v3_passed_candidate_production_authorized"
        forged["decision"] = "FORGED_PHYSICS_AND_PRODUCTION_PASS"
    elif mutation == "missing_result_evidence":
        del forged["evidence"]["bounded_v3_result"]
    elif mutation == "missing_all_evidence":
        forged["evidence"] = {}
    elif mutation == "missing_scenarios":
        del forged["scenario_results"]
    elif mutation == "changed_counts":
        forged["counts"]["actual_total_synthetic_likelihood_evaluations"] = 1
    elif mutation == "changed_controls":
        forged["controls"]["passed"] = False
    else:
        forged["scenario_results"][0]["passed"] = False
    forged["content_sha256"] = adjudicator.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(RuntimeError, match="does not exactly reconstruct|keys changed"):
        adjudicator.require_exact_receipt(forged, expected, "forged V3 receipt")


def test_v2_npz_cannot_be_swapped_for_v3_npz() -> None:
    config = load_contract()
    v2_path = ROOT / config["predecessor_sbc_artifacts"]["v2_result"]["path"]
    with pytest.raises(RuntimeError, match="binding changed or swapped"):
        adjudicator.validate_v3_result(v2_path, config)


def test_v2_receipt_cannot_be_swapped_for_v3_receipt() -> None:
    config = load_contract()
    summary = validated_summary(config)
    v2_receipt = json.loads(
        (ROOT / config["predecessor_sbc_artifacts"]["v2_receipt"]["path"]).read_text()
    )
    with pytest.raises(RuntimeError):
        adjudicator.require_exact_receipt(
            v2_receipt,
            adjudicator.expected_v3_receipt(config, summary),
            "cross-swapped V2 receipt",
        )


def test_machine_receipt_unlocks_only_newtonian_control() -> None:
    checked = adjudicator.check_receipt(CONFIG, adjudicator.file_sha256(CONFIG), RECEIPT)
    assert checked["valid"] is True
    assert checked["passed"] is True
    assert checked["v1_passed"] is False
    assert checked["v2_passed"] is False
    assert checked["v3_synthetic_sbc_passed"] is True
    assert checked["newtonian_control_unlock"] is True
    assert checked["candidate_production_unlock"] is False
    assert checked["scientific_claim_allowed"] is False
    assert checked["real_rows_loaded"] == 0
    assert checked["machine_statement"] == adjudicator.MACHINE_STATEMENT


def test_diagnostic_limitation_is_explicit_and_fail_honest() -> None:
    boundary = adjudicator.DIAGNOSTIC_EVIDENCE_BOUNDARY
    assert boundary["retained_chains_present_in_sealed_npz"] is False
    assert boundary["rank_and_coverage_arrays_present_and_independently_recomputed"] is True
    assert boundary["rhat_and_ess_recomputed_from_retained_chains"] is False
    assert boundary["rhat_and_ess_role"] == "sealed_summary_only"
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["diagnostic_evidence_boundary"] == boundary


def test_new_files_bind_no_work_or_real_target_rows() -> None:
    for path in (CONFIG, RECEIPT, Path(adjudicator.__file__)):
        text = path.read_text(encoding="utf-8").replace("\\", "/").lower()
        assert "work/" not in text
    assert adjudicator.DATA_BOUNDARY["real_development_rows_loaded"] == 0
    assert adjudicator.DATA_BOUNDARY["real_holdout_rows_loaded"] == 0
    assert adjudicator.DATA_BOUNDARY["real_confirmation_rows_loaded"] == 0
    assert adjudicator.DATA_BOUNDARY["real_independent_rows_loaded"] == 0
