from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.constraint_recovered_identity_breadth_report import (
    CLAIMS,
    IPYNB_PATH,
    MARKDOWN_PATH,
    _content_sha,
    build_outputs,
    build_report,
    render_ipynb,
    render_markdown,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return build_report(ROOT)


def _reseal(value: dict) -> None:
    value["content_sha256"] = _content_sha(value)


def test_checked_receipt_binding_and_counts_are_exact(report: dict) -> None:
    binding = report["bindings"]["receipt"]
    receipt = json.loads((ROOT / binding["path"]).read_text(encoding="utf-8"))
    assert receipt["content_sha256"] == binding["content_sha256"]
    assert report["counts"] == {
        "blocked": 0,
        "false_controls_rejected": 1,
        "integer_polynomial_replays": 2,
        "kernel_checked_theorems": 1,
        "kernel_executions": 2,
        "recovered_candidates_bound": 14,
        "recovered_worlds_bound": 2,
        "rejected": 0,
        "symbolic_certificates_bound": 2,
    }


def test_derivations_preserve_exact_integer_replay(report: dict) -> None:
    quartic = report["quartic_derivation"]
    assert quartic["first_product"]["coefficients_constant_first"] == [-6, 1, 1]
    assert quartic["computed_coefficients_constant_first"] == [-30, -1, 0, 2, 1]
    assert (
        quartic["computed_coefficients_constant_first"]
        == quartic["recovered_coefficients_constant_first"]
    )
    partial = report["partial_fraction_derivation"]
    assert partial["computed_numerator_coefficients_constant_first"] == [127, 53, 6]
    assert partial["computed_denominator_coefficients_constant_first"] == [70, 59, 14, 1]
    assert partial["regular_domain_exclusions"] == [-7, -5, -2]
    assert quartic["floating_point_operations"] == 0
    assert partial["floating_point_operations"] == 0


def test_lean_success_and_false_control_are_visible(report: dict) -> None:
    lean = report["lean_check"]
    assert lean["decision"] == "pass_lean_checked_closed_premise"
    assert lean["exit_code"] == 0
    assert lean["dependency_closure_valid"] is True
    assert lean["sorry_or_axiom_used"] is False
    false = report["false_control"]
    assert false == {
        "alteration": "constant coefficient -30 changed to -29",
        "decision": "block_lean_process_failure",
        "nonzero_exit_code": True,
        "rejected_before_receipt_promotion": True,
        "target": "Invariant.constraintRecoveredQuarticFalseControl",
    }


def test_markdown_and_ipynb_are_deterministic_semantic_twins_and_committed(
    report: dict,
) -> None:
    markdown = render_markdown(report)
    notebook = render_ipynb(report)
    assert "## 1. Quartic identity" in markdown
    assert "## 2. Partial-fraction identity" in markdown
    assert "## 3. Independent Lean kernel check" in markdown
    assert "## 4. Deliberate failure" in markdown
    assert "does not establish general formula discovery" in markdown
    assert "".join(notebook["cells"][0]["source"]) == markdown
    assert notebook["metadata"]["report_content_sha256"] == report["content_sha256"]
    first = build_outputs(ROOT)
    second = build_outputs(ROOT)
    assert first == second
    assert (ROOT / MARKDOWN_PATH).read_text(encoding="utf-8") == first[0]
    assert (ROOT / IPYNB_PATH).read_text(encoding="utf-8") == first[1]
    assert (
        hashlib.sha256(first[0].encode()).hexdigest()
        == hashlib.sha256((ROOT / MARKDOWN_PATH).read_bytes()).hexdigest()
    )


def test_bounded_claims_are_explicit(report: dict) -> None:
    assert report["claims"] == CLAIMS
    assert CLAIMS["checked_receipt_native_validated"] is True
    assert CLAIMS["general_formula_discovery_established"] is False
    assert CLAIMS["novelty_established"] is False
    assert CLAIMS["promotion_authorized"] is False
    assert CLAIMS["scientific_or_physics_truth_inferred"] is False


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda value: value["quartic_derivation"].__setitem__(
                "computed_coefficients_constant_first", [-29, -1, 0, 2, 1]
            ),
            "semantic boundary",
        ),
        (
            lambda value: value["partial_fraction_derivation"].__setitem__(
                "regular_domain_exclusions", []
            ),
            "semantic boundary",
        ),
        (
            lambda value: value["lean_check"].__setitem__("exit_code", 1),
            "semantic boundary",
        ),
        (
            lambda value: value["false_control"].__setitem__("nonzero_exit_code", False),
            "semantic boundary",
        ),
        (
            lambda value: value["claims"].__setitem__(
                "general_formula_discovery_established", True
            ),
            "claim boundary",
        ),
        (
            lambda value: value.__setitem__("unknown_top_level_key", True),
            "schema or seal",
        ),
    ],
)
def test_resealed_tampers_fail_closed(report: dict, mutator, message: str) -> None:
    tampered = copy.deepcopy(report)
    mutator(tampered)
    _reseal(tampered)
    with pytest.raises(ValueError, match=message):
        validate_report(tampered)
