from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.alpha_observational_rehearsal import (
    build_observational_rehearsal,
)
from sigma_theory_compiler.alpha_operational_rehearsal import run_operational_rehearsal
from sigma_theory_compiler.comprehensive_alpha_coverage_report import (
    OUTPUT_PATH,
    build_markdown,
    derive_coverage,
    render_markdown,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def operational(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return run_operational_rehearsal(
        ROOT, tmp_path_factory.mktemp("alpha-coverage-operational") / "owned"
    )


@pytest.fixture(scope="module")
def observational() -> dict[str, object]:
    return build_observational_rehearsal(ROOT)


@pytest.fixture(scope="module")
def report(operational: dict[str, object], observational: dict[str, object]) -> dict[str, object]:
    return derive_coverage(ROOT, operational, observational)


def test_matrix_is_derived_from_validated_receipts(report: dict[str, object]) -> None:
    assert set(report["inputs"]) == {
        "cross_generator",
        "curriculum",
        "gpu_stress",
        "human_record",
        "lean_kernel",
        "observational_rehearsal",
        "operational_rehearsal",
    }
    assert report["counts"] == {
        "vertical_slices": 8,
        "exercised": 8,
        "partial": 0,
        "blocked": 0,
        "curriculum_ready": 5,
        "curriculum_missing": 195,
    }
    assert [(row["ordinal"], row["status"]) for row in report["slices"]] == [
        (1, "exercised"),
        (2, "exercised"),
        (3, "exercised"),
        (4, "exercised"),
        (5, "exercised"),
        (6, "exercised"),
        (7, "exercised"),
        (8, "exercised"),
    ]
    lean = report["slices"][2]
    assert "kernel-checked 1 bounded theorem" in lean["evidence"]
    assert "known-answer protocol smoke theorem only" in lean["boundary"]
    assert "post-alpha slots remain missing" in report["slices"][3]["boundary"]
    assert "attempt-2 completion" in report["slices"][4]["evidence"]
    accelerator = report["slices"][5]
    assert "5,341,184 GPU/CPU comparisons" in accelerator["evidence"]
    assert "87,509,958,656 measured synthetic evaluations" in accelerator["evidence"]
    assert "no proof, ranking, observations" in accelerator["boundary"]
    human = report["inputs"]["human_record"]
    assert human["identity_path"].endswith("math-known-identity-pipeline-control/campaign.json")
    assert human["notebook_md_path"].endswith("natural-sum-rediscovery.md")
    assert human["notebook_ipynb_path"].endswith("natural-sum-rediscovery.ipynb")
    assert human["walkthrough_path"].endswith("KNOWN_ANSWER_SUCCESS_FAILURE_WALKTHROUGH.md")
    assert report["claims"] == {
        "receipt_projection_validated": True,
        "comprehensive_alpha_exit_reached": False,
        "scientific_truth_established": False,
        "novelty_established": False,
        "promotion_authorized": False,
        "curriculum_complete": False,
    }


def test_success_block_reject_error_narratives_preserve_receipt_outcomes(
    report: dict[str, object],
) -> None:
    assert report["generator_dispositions"] == {
        "bayesian": "pass",
        "cross_domain": "block",
        "egraph": "reject",
        "evolutionary": "pass",
        "grammar": "pass",
        "llm": "error",
        "symbolic": "pass",
    }
    markdown = render_markdown(report)
    assert "### Success" in markdown
    assert "`bayesian, evolutionary, grammar, symbolic` passed both" in markdown
    assert "`cross_domain` passed `hard_exact` but was blocked" in markdown
    assert "`egraph` was rejected at `hard_exact`" in markdown
    assert "`llm` is a quarantined offline proposal" in markdown
    assert "operational LLM adapter was blocked before any provider call" in markdown
    assert "## Historical accelerator control" in markdown
    assert "## Human-readable record" in markdown
    assert "**5/200**" in markdown and "**195 slots remain missing**" in markdown


def test_generated_markdown_is_byte_deterministic_and_committed(
    operational: dict[str, object], observational: dict[str, object]
) -> None:
    first = build_markdown(ROOT, operational, observational)
    second = build_markdown(
        ROOT,
        json.loads(json.dumps(operational)),
        json.loads(json.dumps(observational)),
    )
    assert first == second
    assert (ROOT / OUTPUT_PATH).read_text(encoding="utf-8") == first
    forbidden = (
        "scientific truth established: true",
        "novelty established: true",
        "promotion authorized: true",
        "comprehensive-alpha exit gate is reached",
    )
    assert not any(text in first.lower() for text in forbidden)


def test_report_and_operational_tamper_fail_closed(
    report: dict[str, object],
    operational: dict[str, object],
    observational: dict[str, object],
) -> None:
    forged_report = copy.deepcopy(report)
    forged_report["claims"]["comprehensive_alpha_exit_reached"] = True
    with pytest.raises(ValueError, match="self-seal changed"):
        render_markdown(forged_report)

    resealed = copy.deepcopy(report)
    resealed["slices"][2]["status"] = "blocked"
    resealed["content_sha256"] = canonical_sha256(
        {key: value for key, value in resealed.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="semantic contract changed"):
        render_markdown(resealed)

    forged_operational = copy.deepcopy(operational)
    forged_operational["claims"]["scientific_pass"] = True
    with pytest.raises(ValueError, match="hash or schema mismatch"):
        derive_coverage(ROOT, forged_operational, observational)

    forged_observational = copy.deepcopy(observational)
    forged_observational["claims"]["scientific_pass"] = True
    with pytest.raises(ValueError, match="hash or schema mismatch"):
        derive_coverage(ROOT, operational, forged_observational)
