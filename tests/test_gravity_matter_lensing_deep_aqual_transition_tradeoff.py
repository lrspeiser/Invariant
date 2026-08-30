from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_deep_aqual_transition_tradeoff as tradeoff

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_predecessors_are_exact() -> None:
    config = tradeoff.load_config(ROOT)
    results = tradeoff.validate_predecessors(ROOT, config)
    assert [item["binding_id"] for item in results] == [
        "external_metric_principal_symbol",
        "scalar_hamiltonian_necessary_conditions",
    ]
    assert all(item["all_current_and_commit_hashes_match"] for item in results)


def test_symbolic_transition_theorem_and_regulator_checks_pass() -> None:
    checks, formulas = tradeoff.symbolic_checks()
    assert len(checks) == 24
    assert all(item["passed"] for item in checks)
    assert tuple(item["check_id"] for item in checks) == tradeoff.SYMBOLIC_CHECK_IDS
    assert formulas["exact_K"] == "(1+2*p)*A*(-X)^p"
    assert formulas["regulated_timelike_K"].endswith("^(3/2)")


def test_numeric_tradeoff_cases_pass_and_preserve_cone_warning() -> None:
    results = tradeoff.numeric_checks(tradeoff.load_config(ROOT))
    assert len(results) == 4
    assert all(item["passed"] for item in results)
    timelike = next(item for item in results if item["case_id"].startswith("TIMELIKE"))
    assert timelike["superluminal_relative_conformal_cone"] is True
    assert timelike["speed_squared"] == pytest.approx(2.0)
    deep = next(item for item in results if item["case_id"] == "SPACELIKE_DEEP_APPROXIMATION")
    assert deep["aqual_relative_error"] == pytest.approx(0.004987562112089027)


def test_receipt_claim_ceiling_is_partial_and_zero_access() -> None:
    receipt = tradeoff.build_receipt(ROOT)
    assert receipt["decision"] == tradeoff.DECISION
    assert receipt["counts"]["symbolic_checks_passed"] == 24
    assert receipt["counts"]["numeric_cases_passed"] == 4
    assert receipt["adjudication"]["exact_deep_aqual_transition_conditional_no_go_derived"] is True
    assert receipt["adjudication"]["positive_floor_regulator_removes_transition_degeneracy"] is True
    assert (
        receipt["adjudication"]["positive_floor_regulator_preserves_exact_low_gradient_aqual"]
        is False
    )
    assert (
        receipt["adjudication"]["regulated_example_is_subluminal_relative_to_conformal_matter_cone"]
        is False
    )
    assert receipt["adjudication"]["CP11_4_complete"] is False
    assert receipt["claim_boundary"]["healthy_action_established"] is False
    assert receipt["claim_boundary"]["scientific_observational_claim_allowed"] is False
    assert set(receipt["zero_access_and_compute"].values()) == {0}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["adjudication"].__setitem__("CP11_4_complete", True),
        lambda value: value["exact_transition_theorem"].__setitem__(
            "conditional_no_go", "unconditional no-go"
        ),
        lambda value: value["regulated_example_contract"].__setitem__(
            "timelike_cone_warning", "none"
        ),
        lambda value: value["zero_access_and_compute"].__setitem__("network_calls", 1),
    ],
)
def test_config_mutations_fail_closed(mutation: object) -> None:
    config = tradeoff.load_config(ROOT)
    changed = copy.deepcopy(config)
    mutation(changed)  # type: ignore[operator]
    with pytest.raises(tradeoff.DeepAqualTransitionError, match="content changed"):
        tradeoff.validate_config(changed)


def test_stored_receipt_rebuilds_exactly() -> None:
    path = ROOT / tradeoff.OUTPUT_PATH
    stored = json.loads(path.read_text(encoding="utf-8"))
    tradeoff.validate_receipt(stored, ROOT)
    assert stored == tradeoff.build_receipt(ROOT)


def test_atomic_writer_refuses_different_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert tradeoff._atomic_no_clobber(path, b"first\n") == "CREATED"
    assert tradeoff._atomic_no_clobber(path, b"first\n") == "EXISTING_IDENTICAL"
    assert tradeoff._atomic_no_clobber(path, b"second\n") == "EXISTING_DIFFERENT"
    assert path.read_bytes() == b"first\n"
