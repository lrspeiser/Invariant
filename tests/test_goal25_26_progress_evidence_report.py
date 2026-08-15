from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.goal25_26_progress_evidence_report import (
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


def test_eight_receipts_are_natively_validated_and_bound(report: dict) -> None:
    assert report["summary_counts"]["native_receipts_validated"] == 8
    assert len(report["bindings"]) == 8
    for binding in report["bindings"].values():
        receipt = json.loads((ROOT / binding["path"]).read_text(encoding="utf-8"))
        assert receipt["content_sha256"] == binding["content_sha256"]
        assert len(binding["file_sha256"]) == 64


def test_failure_construction_success_counts_are_exact(report: dict) -> None:
    h7 = report["h7_lane"]
    assert h7["failure_boundary"]["exact_direction_controls_passed"] == 6
    assert h7["construction"] == {
        "authorized_but_initially_unregistered_k55_packets": 15,
        "k0_nonzero_entries": 847,
        "k0_normal_form_terms": 2732,
        "sphere_identity_nonzero_remainders": 0,
    }
    assert h7["bounded_success"]["k55_order_one_packets_registered"] == 15
    assert h7["bounded_success"]["tc2_order_one_packets_registered"] == 15
    assert h7["bounded_success"]["differentiated_identity_nonzero_remainders"] == 0
    assert h7["bounded_success"]["tc2_product_rule_nonzero_remainders"] == 0


def test_matter_failure_and_bounded_success_are_exact(report: dict) -> None:
    matter = report["matter_lane"]
    assert matter["first_success"]["symmetry_residual_nonzero_entries"] == 0
    assert matter["typed_failure"]["missing_primitive_slots"] == 780
    assert matter["construction"] == {
        "divergence_components": 4,
        "formal_external_jet_atoms": 580,
        "indexed_formula_templates": 17,
        "physical_metric_operator_slots": 200,
        "primitive_slots_registered": 780,
    }
    assert matter["bounded_success"] == {
        "candidate_flat_row_manifests": 12,
        "flat_constraint_rows_expanded": 4,
        "nonzero_exact_q_sqrt2_coefficients": 112,
    }
    assert report["typed_blockers"][1]["exact_scalar_values_before_domain"] == 1010


def test_markdown_and_notebook_are_deterministic_semantic_twins(report: dict) -> None:
    markdown = render_markdown(report)
    notebook = render_ipynb(report)
    assert "### Failure boundary" in markdown
    assert "### Construction" in markdown
    assert "### Bounded success" in markdown
    assert "do **not** establish full D4" in markdown
    assert "".join(notebook["cells"][0]["source"]) == markdown
    assert notebook["metadata"]["report_content_sha256"] == report["content_sha256"]
    assert build_outputs(ROOT) == build_outputs(ROOT)
    expected_markdown, expected_notebook = build_outputs(ROOT)
    assert (ROOT / MARKDOWN_PATH).read_text(encoding="utf-8") == expected_markdown
    assert (ROOT / IPYNB_PATH).read_text(encoding="utf-8") == expected_notebook


def test_claim_boundary_is_explicit(report: dict) -> None:
    assert report["claims"] == CLAIMS
    assert CLAIMS["full_d4_closed"] is False
    assert CLAIMS["global_h7_closed"] is False
    assert CLAIMS["constraint_propagation_closed"] is False
    assert CLAIMS["universal_all_matter_closure_established"] is False
    assert CLAIMS["promotion_authorized"] is False


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda value: value["matter_lane"]["construction"].__setitem__(
                "indexed_formula_templates", 16
            ),
            "semantic boundary",
        ),
        (
            lambda value: value["h7_lane"]["bounded_success"].__setitem__(
                "differentiated_identity_nonzero_remainders", 1
            ),
            "semantic boundary",
        ),
        (
            lambda value: value["claims"].__setitem__("global_h7_closed", True),
            "claim boundary",
        ),
        (
            lambda value: value.__setitem__("unexpected", True),
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
