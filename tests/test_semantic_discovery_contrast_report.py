from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.constraint_conditioned_semantic_recovery_tournament import (
    validate_campaign as validate_recovery_campaign,
)
from sigma_theory_compiler.semantic_discovery_contrast_report import (
    BLIND_PATH,
    CLAIMS,
    IPYNB_PATH,
    MARKDOWN_PATH,
    RECOVERY_PATH,
    build_outputs,
    build_report,
    render_ipynb,
    render_markdown,
    validate_output_pair,
    validate_report,
)
from sigma_theory_compiler.semantic_formula_proof_holdout_tournament import (
    validate_campaign as validate_blind_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return build_report(ROOT)


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: child for key, child in value.items() if key != "content_sha256"}
    )


def test_both_receipts_are_native_validated_and_byte_bound(
    report: dict[str, object],
) -> None:
    bindings = report["bindings"]
    for name in ("blind_receipt", "recovery_receipt"):
        binding = bindings[name]
        path = ROOT / binding["path"]
        receipt = json.loads(path.read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["file_sha256"]
        assert receipt["content_sha256"] == binding["content_sha256"]
        assert binding["source_bindings"] == receipt["source_bindings"]
    validate_blind_campaign(json.loads((ROOT / BLIND_PATH).read_text(encoding="utf-8")), ROOT)
    validate_recovery_campaign(json.loads((ROOT / RECOVERY_PATH).read_text(encoding="utf-8")), ROOT)


def test_exact_contrast_and_prospective_chronology_are_receipt_derived(
    report: dict[str, object],
) -> None:
    assert report["contrast"] == {
        "blind": {
            "worlds": 3,
            "native_generator_families": 7,
            "structured_candidates": 21,
            "passes": 0,
            "rejects": 21,
            "blocks": 0,
            "exact_counterexamples": 21,
            "pareto_eligible_candidates": 0,
        },
        "constraint_conditioned": {
            "worlds": 3,
            "native_generator_families": 7,
            "recovered_candidates": 21,
            "passes": 21,
            "rejects": 0,
            "blocks": 0,
            "proof_certificates": 21,
            "pareto_eligible_candidates": 21,
        },
        "paired_cohorts_use_distinct_hidden_worlds": True,
    }
    chronology = {
        "generation_events_before_unseal": 21,
        "pre_unseal_target_access_count": 0,
        "atomic_unseal_batches": 1,
        "target_records_unsealed": 3,
        "post_unseal_generation_count": 0,
        "post_unseal_tuning_events": 0,
    }
    assert report["chronology"] == {
        "blind": chronology,
        "constraint_conditioned": chronology,
    }


def test_public_evidence_generic_solver_certificates_and_controls_are_exact(
    report: dict[str, object],
) -> None:
    public = report["public_evidence"]
    assert [row["basis_term_count"] for row in public["worlds"]] == [5, 3, 4]
    assert [row["evidence_count"] for row in public["worlds"]] == [5, 3, 2]
    assert all(row["closed_form_was_public"] is False for row in public["worlds"])
    serialized_public = json.dumps(public, sort_keys=True)
    for hidden_formula in (
        "x**4 + 2*x**3 - x - 30",
        "3/(x + 2) - 2/(x + 5) + 5/(x + 7)",
        "2*n**3 + 2*n**2 + n + 7",
    ):
        assert hidden_formula not in serialized_public

    solver = report["generic_solver"]
    assert solver["synthesis_invocations"] == 21
    assert solver["lineage_role"] == "lineage_and_seed_proposal_only"
    assert all(
        row["lineage_candidates"] == 7 and row["unique_recovered_expressions"] == 1
        for row in solver["unique_expression_counts"]
    )
    assert report["certificates"]["blind_candidate_exact_counterexamples"] == 21
    assert sum(report["certificates"]["recovery_candidate_certificate_schemas"].values()) == 21
    assert [
        (row["expected_outcome"], row["observed_outcome"]) for row in report["negative_controls"]
    ] == [
        ("BLOCK", "BLOCK"),
        ("BLOCK", "BLOCK"),
        ("REJECT", "REJECT"),
    ]


def test_claim_boundary_explicitly_denies_native_discovery_and_broad_promotion(
    report: dict[str, object],
) -> None:
    assert report["claims"] == CLAIMS
    assert (
        report["claims"]["native_generators_independently_discovered_recovered_formulas"] is False
    )
    assert report["claims"]["contrast_is_a_matched_target_performance_comparison"] is False
    assert report["claims"]["constraint_recovery_establishes_general_discovery"] is False
    assert report["claims"]["scientific_truth_established"] is False
    assert report["claims"]["novelty_established"] is False
    assert report["claims"]["promotion_authorized"] is False


def test_markdown_and_notebook_are_deterministic_committed_semantic_twins(
    report: dict[str, object],
) -> None:
    markdown = render_markdown(report)
    notebook = render_ipynb(report)
    assert "0 | 21 | 0" in markdown
    assert "21 | 0 | 0" in markdown
    assert "not a matched-target scorecard" in markdown
    assert "not to seven independent discoveries" in markdown
    assert "rank([A|b])" in markdown
    assert "".join(notebook["cells"][0]["source"]) == markdown
    assert notebook["metadata"]["report_content_sha256"] == report["content_sha256"]
    outputs = build_outputs(ROOT)
    assert outputs == build_outputs(ROOT)
    assert (ROOT / MARKDOWN_PATH).read_text(encoding="utf-8") == outputs[0]
    assert (ROOT / IPYNB_PATH).read_text(encoding="utf-8") == outputs[1]
    validate_output_pair(report, *outputs)


def test_resealed_receipt_report_and_output_tampers_fail_closed(
    report: dict[str, object],
) -> None:
    blind = json.loads((ROOT / BLIND_PATH).read_text(encoding="utf-8"))
    blind["counts"]["candidate_passes"] = 1
    _reseal(blind)
    with pytest.raises(ValueError, match="exact replay mismatch"):
        validate_blind_campaign(blind, ROOT)

    recovery = json.loads((ROOT / RECOVERY_PATH).read_text(encoding="utf-8"))
    recovery["counts"]["candidate_passes"] = 20
    _reseal(recovery)
    with pytest.raises(ValueError, match="exact replay mismatch"):
        validate_recovery_campaign(recovery, ROOT)

    promoted = copy.deepcopy(report)
    promoted["claims"]["promotion_authorized"] = True
    _reseal(promoted)
    with pytest.raises(ValueError, match="claim boundary changed"):
        validate_report(promoted)

    markdown = render_markdown(report)
    notebook = json.dumps(render_ipynb(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with pytest.raises(ValueError, match="output pair changed"):
        validate_output_pair(report, markdown + "tampered\n", notebook)
    tampered_notebook = json.loads(notebook)
    tampered_notebook["cells"][0]["source"].append("tampered\n")
    with pytest.raises(ValueError, match="output pair changed"):
        validate_output_pair(
            report,
            markdown,
            json.dumps(tampered_notebook, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
