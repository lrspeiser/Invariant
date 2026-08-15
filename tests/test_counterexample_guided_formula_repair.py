from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.counterexample_guided_formula_repair import (
    CounterexampleGuidedRepairError,
    build_checked_receipt,
    build_repair_trace,
    load_config,
    validate_checked_receipt,
    validate_repair_trace,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "counterexample_guided_formula_repair.json"
RECEIPT = ROOT / "runs" / "math" / "counterexample-guided-formula-repair" / "receipt.json"


def _config() -> dict:
    return load_config(CONFIG)


def _reseal(value: dict) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_counterexample_is_consumed_and_round_two_passes() -> None:
    trace = build_repair_trace(_config())
    validate_repair_trace(trace, _config())
    assert trace["decision"] == "PASS"
    assert [item["decision"] for item in trace["rounds"]] == ["REJECT", "PASS"]
    assert trace["rounds"][0]["reason_codes"] == ["heldout_counterexample"]
    witness = trace["counterexample_constraint"]
    assert witness["point"] == {"numerator": 2, "denominator": 1}
    assert witness["value"] == {"numerator": 7, "denominator": 1}
    assert trace["rounds"][1]["consumed_witness_sha256"] == witness["witness_sha256"]
    assert trace["rounds"][1]["synthesis_rows"] == 3
    assert trace["rounds"][1]["validation_rows_checked"] == 2
    assert trace["final_candidate"]["representation"]["expression"] == "x**2 + x + 1"
    assert trace["final_translation"]["translation_kind"] == (
        "integer_polynomial_coefficient_identity"
    )
    assert trace["counts"] == {
        "rounds_executed": 2,
        "counterexamples_consumed": 1,
        "basis_terms_added": 1,
        "final_candidates_passed": 1,
        "kernel_checks_executed": 0,
    }


def test_checked_artifact_executes_real_lean_and_replays() -> None:
    checked = json.loads(RECEIPT.read_text(encoding="utf-8"))
    validate_checked_receipt(checked, _config())
    assert checked["decision"] == "PASS"
    assert checked["kernel_check"]["decision"] == "pass_lean_checked_closed_premise"
    assert checked["kernel_check"]["dependency_audit"]["closure_valid"] is True
    assert checked["claims"]["lean_kernel_executed"] is True
    live = build_checked_receipt(_config())
    assert live == checked


def test_no_solution_control_terminates_block_after_two_rounds() -> None:
    config = _config()
    # x^3 + x + 1: the declared quadratic repair cannot satisfy the remaining holdout.
    for row in config["initial_problem"]["constraints"]["rows"]:
        x = row["point"]["numerator"]
        row["value"] = {"numerator": x**3 + x + 1, "denominator": 1}
    for row in config["initial_problem"]["validation"]["rows"]:
        x = row["point"]["numerator"]
        row["value"] = {"numerator": x**3 + x + 1, "denominator": 1}
    trace = build_repair_trace(config)
    assert trace["decision"] == "BLOCK"
    assert trace["reason_codes"] == ["repair_budget_exhausted"]
    assert [item["decision"] for item in trace["rounds"]] == ["REJECT", "REJECT"]
    assert trace["counts"]["rounds_executed"] == 2
    assert trace["final_candidate"] is None
    assert trace["final_translation"] is None


def test_resealed_witness_chronology_and_candidate_tamper_fail() -> None:
    config = _config()
    trace = build_repair_trace(config)
    for mutate in (
        lambda value: value["counterexample_constraint"].__setitem__(
            "value", {"numerator": 8, "denominator": 1}
        ),
        lambda value: value["rounds"][1].__setitem__("consumed_witness_sha256", "0" * 64),
        lambda value: value["final_candidate"]["representation"].__setitem__(
            "expression", "x**2 + x + 2"
        ),
    ):
        tampered = copy.deepcopy(trace)
        mutate(tampered)
        _reseal(tampered)
        with pytest.raises(CounterexampleGuidedRepairError):
            validate_repair_trace(tampered, config)


def test_checked_receipt_kernel_and_claim_tamper_fail() -> None:
    config = _config()
    checked = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for mutate in (
        lambda value: value["kernel_check"].__setitem__("source_sha256", "0" * 64),
        lambda value: value["claims"].__setitem__("general_repair_completeness", True),
    ):
        tampered = copy.deepcopy(checked)
        mutate(tampered)
        if "kernel_check" in tampered:
            _reseal(tampered["kernel_check"])
        _reseal(tampered)
        with pytest.raises(CounterexampleGuidedRepairError):
            validate_checked_receipt(tampered, config)


def test_config_rejects_unbounded_rounds_and_duplicate_basis() -> None:
    config = _config()
    config["max_rounds"] = 3
    with pytest.raises(CounterexampleGuidedRepairError):
        build_repair_trace(config)
    config = _config()
    config["repair_basis_append"] = ["x"]
    with pytest.raises(CounterexampleGuidedRepairError):
        build_repair_trace(config)
