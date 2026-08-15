from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.prospective_tournament_research_report import (
    CLAIMS,
    IPYNB_PATH,
    MARKDOWN_PATH,
    build_outputs,
    build_report,
    render_ipynb,
    render_markdown,
    validate_report,
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


def test_bindings_validate_exact_tournament_robustness_and_prior_art(
    report: dict[str, object],
) -> None:
    bindings = report["bindings"]
    for binding in (bindings["tournament"], bindings["robustness"]):
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["file_sha256"]
        assert (
            json.loads(path.read_text(encoding="utf-8"))["content_sha256"]
            == binding["content_sha256"]
        )
    prior = bindings["prior_art"]
    for prefix in ("seed", "policy", "audit"):
        assert (
            hashlib.sha256((ROOT / prior[f"{prefix}_path"]).read_bytes()).hexdigest()
            == prior[f"{prefix}_file_sha256"]
        )
    assert report["prior_art"]["corpus_counts"] == {
        "artifacts": 18,
        "derivations": 3,
        "edges": 36,
        "equations": 18,
        "equivalence_edges": 1,
        "nodes": 31,
        "sources": 9,
    }


def test_chronology_pass_and_honest_reject_are_exact(report: dict[str, object]) -> None:
    assert report["chronology"] == {
        "preregistered_worlds": 3,
        "native_generator_families_per_world": 7,
        "generation_events_before_unseal": 21,
        "pre_unseal_target_access_count": 0,
        "atomic_unseal_batches": 1,
        "target_records_unsealed": 3,
        "post_unseal_generation_count": 0,
        "post_unseal_tuning_events": 0,
    }
    assert report["representative_pass"]["world_id"] == "prospective.modular_affine"
    assert report["representative_pass"]["target_hypothesis"] == 7
    assert report["representative_pass"]["surviving_families"] == ["bayesian"]
    rejection = report["fixed_budget_reject"]
    assert rejection["world_id"] == "prospective.finite_difference"
    assert rejection["target_hypothesis"] == 1
    assert rejection["candidate_hypotheses"] == {
        "bayesian": 10,
        "cross_domain": 5,
        "egraph": 0,
        "evolutionary": 5,
        "grammar": 3,
        "llm": 0,
        "symbolic": 9,
    }
    assert rejection["surviving_families"] == []


def test_robustness_dependence_and_corpus_boundary_do_not_overclaim(
    report: dict[str, object],
) -> None:
    assert report["robustness"] == {
        "registered_seeds": 8,
        "cpu_replay_passes": 8,
        "candidate_evaluation_replays": 168,
        "gate_outcome_comparisons": 336,
        "pareto_recomputations": 28,
        "candidate_overlap_per_seed": {
            "intersection": 21,
            "union": 21,
            "jaccard": {"numerator": 1, "denominator": 1},
        },
        "gate_status_stable": True,
        "fronts_stable": True,
    }
    assert report["ablation"] == {
        "dependent_family": "bayesian",
        "pass_to_reject_worlds": 2,
        "front_members_removed": 2,
        "other_family_decision_changes": 0,
        "other_family_front_changes": 0,
    }
    assert report["prior_art"]["candidate_queries"] == 21
    assert report["prior_art"]["status_counts"] == {"absent_from_this_corpus": 21}
    assert report["claims"] == CLAIMS
    assert report["claims"]["corpus_absence_establishes_novelty"] is False
    assert report["claims"]["general_discovery_established"] is False


def test_markdown_and_ipynb_are_byte_deterministic_semantic_twins_and_committed(
    report: dict[str, object],
) -> None:
    markdown = render_markdown(report)
    notebook = render_ipynb(report)
    assert "## The honest fixed-budget REJECT" in markdown
    assert "Bayesian-dependent, not portfolio-robust" in markdown
    assert "corpus absence is not novelty" in markdown.lower()
    assert "not evidence that new search seeds discover the same mathematics" in markdown
    assert "".join(notebook["cells"][0]["source"]) == markdown
    assert notebook["metadata"]["report_content_sha256"] == report["content_sha256"]
    first = build_outputs(ROOT)
    second = build_outputs(ROOT)
    assert first == second
    assert (ROOT / MARKDOWN_PATH).read_text(encoding="utf-8") == first[0]
    assert (ROOT / IPYNB_PATH).read_text(encoding="utf-8") == first[1]


def test_resealed_chronology_novelty_and_prior_art_tampers_fail_closed(
    report: dict[str, object],
) -> None:
    chronology = copy.deepcopy(report)
    chronology["chronology"]["pre_unseal_target_access_count"] = 1
    _reseal(chronology)
    with pytest.raises(ValueError, match="semantic boundary changed"):
        validate_report(chronology)

    novelty = copy.deepcopy(report)
    novelty["claims"]["corpus_absence_establishes_novelty"] = True
    _reseal(novelty)
    with pytest.raises(ValueError, match="claim boundary changed"):
        validate_report(novelty)

    corpus = copy.deepcopy(report)
    corpus["prior_art"]["status_counts"] = {"present_in_this_corpus": 21}
    _reseal(corpus)
    with pytest.raises(ValueError, match="semantic boundary changed"):
        validate_report(corpus)
