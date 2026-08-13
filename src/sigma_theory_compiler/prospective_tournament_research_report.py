"""Receipt-derived Markdown and notebook for the prospective tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .equation_universe_knowledge_adapter import (
    EquationUniverseKnowledgeImport,
    import_equation_universe_files,
)
from .prospective_blind_cross_generator_tournament import (
    validate_campaign as validate_tournament,
)
from .prospective_tournament_robustness_ablation import (
    validate_campaign as validate_robustness,
)
from .sigma_core import canonical_sha256

REPORT_SCHEMA = "sigma-prospective-tournament-research-report-1.0"
TOURNAMENT_PATH = "runs/math/prospective-blind-cross-generator-tournament/campaign.json"
ROBUSTNESS_PATH = "runs/math/prospective-tournament-robustness-ablation/campaign.json"
PRIOR_ART_SEED_PATH = "configs/equation_universe/gravity_seed_v1.json"
PRIOR_ART_POLICY_PATH = "configs/equation_universe/source_policy.json"
PRIOR_ART_AUDIT_PATH = "runs/equation-universe/audit-report.json"
SOURCE_PATH = "src/sigma_theory_compiler/prospective_tournament_research_report.py"
TEST_PATH = "tests/test_prospective_tournament_research_report.py"
MARKDOWN_PATH = "docs/notebooks/generated/prospective-tournament-robustness.md"
IPYNB_PATH = "docs/notebooks/generated/prospective-tournament-robustness.ipynb"
CLAIMS = {
    "receipt_projection_validated": True,
    "chronology_proves_no_pre_unseal_target_access": True,
    "one_world_pass_is_bounded_search_success": True,
    "fixed_budget_reject_is_honest_terminal_evidence": True,
    "bayesian_only_dependence_identified": True,
    "corpus_absence_establishes_novelty": False,
    "stability_establishes_truth": False,
    "promotion_authorized": False,
    "general_discovery_established": False,
}


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("report path is not portable")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("report path escapes project root") from error
    return resolved


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("research-report input must be an object")
    return value


def _candidate_hypotheses(world: Mapping[str, Any]) -> dict[str, int]:
    candidates = {row["artifact_id"]: row for row in world["candidates"]}
    return {
        binding["family"]: candidates[binding["candidate"]["artifact_id"]]["representation"][
            "hypothesis"
        ]
        for binding in world["family_bindings"]
    }


def build_report(root: Path) -> dict[str, Any]:
    """Validate every source and derive the closed human-readable projection."""

    root = root.resolve()
    tournament_path = _resolve(root, TOURNAMENT_PATH)
    robustness_path = _resolve(root, ROBUSTNESS_PATH)
    tournament = _load_json(tournament_path)
    robustness = _load_json(robustness_path)
    validate_tournament(tournament, root)
    validate_robustness(robustness, root)
    prior_art = import_equation_universe_files(
        _resolve(root, PRIOR_ART_SEED_PATH),
        _resolve(root, PRIOR_ART_POLICY_PATH),
        _resolve(root, PRIOR_ART_AUDIT_PATH),
        project_root=root,
    )
    EquationUniverseKnowledgeImport.from_dict(prior_art.to_dict())

    phase = tournament["phase_ledger"]
    if (
        phase["generation_events_before_first_unseal"] != 21
        or phase["pre_unseal_target_access_count"] != 0
        or phase["target_unseal_batches"] != 1
        or phase["post_unseal_generation_count"] != 0
        or phase["post_unseal_tuning_events"] != 0
    ):
        raise ValueError("tournament chronology changed")
    passes = [
        world
        for world in tournament["world_results"]
        if world["decision"] == "pass_at_least_one_target_blind_candidate_survived"
    ]
    rejects = [
        world
        for world in tournament["world_results"]
        if world["decision"] == "reject_fixed_budget_exhausted_without_holdout_match"
    ]
    if len(passes) != 2 or len(rejects) != 1:
        raise ValueError("tournament PASS/REJECT distribution changed")
    pass_world = next(
        world for world in passes if world["world_id"] == "prospective.modular_affine"
    )
    reject_world = rejects[0]
    if reject_world["world_id"] != "prospective.finite_difference":
        raise ValueError("representative fixed-budget REJECT changed")
    bayesian_ablation = next(
        row for row in robustness["ablations"] if row["removed_family"] == "bayesian"
    )
    other_ablations = [
        row for row in robustness["ablations"] if row["removed_family"] != "bayesian"
    ]
    if (
        bayesian_ablation["world_pass_to_reject_count"] != 2
        or bayesian_ablation["front_assignment_change_count"] != 2
        or any(
            row["world_pass_to_reject_count"] != 0 or row["front_assignment_change_count"] != 0
            for row in other_ablations
        )
    ):
        raise ValueError("generator-dependence result changed")

    candidate_hashes = [
        candidate["content_sha256"]
        for world in tournament["world_results"]
        for candidate in world["candidates"]
    ]
    corpus_statuses = Counter(
        prior_art.graph.lookup_content(content_sha256).status.value
        for content_sha256 in candidate_hashes
    )
    if corpus_statuses != {"absent_from_this_corpus": 21}:
        raise ValueError("prior-art corpus lookup result changed")

    body = {
        "schema_version": REPORT_SCHEMA,
        "bindings": {
            "tournament": {
                "path": TOURNAMENT_PATH,
                "file_sha256": _file_sha(tournament_path),
                "content_sha256": tournament["content_sha256"],
            },
            "robustness": {
                "path": ROBUSTNESS_PATH,
                "file_sha256": _file_sha(robustness_path),
                "content_sha256": robustness["content_sha256"],
            },
            "prior_art": {
                "seed_path": PRIOR_ART_SEED_PATH,
                "seed_file_sha256": _file_sha(_resolve(root, PRIOR_ART_SEED_PATH)),
                "policy_path": PRIOR_ART_POLICY_PATH,
                "policy_file_sha256": _file_sha(_resolve(root, PRIOR_ART_POLICY_PATH)),
                "audit_path": PRIOR_ART_AUDIT_PATH,
                "audit_file_sha256": _file_sha(_resolve(root, PRIOR_ART_AUDIT_PATH)),
                "import_content_sha256": prior_art.content_sha256,
                "graph_content_sha256": prior_art.graph.content_sha256,
            },
            "builder": {"path": SOURCE_PATH, "file_sha256": _file_sha(_resolve(root, SOURCE_PATH))},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_resolve(root, TEST_PATH))},
        },
        "chronology": {
            "preregistered_worlds": tournament["counts"]["worlds"],
            "native_generator_families_per_world": tournament["counts"]["generator_families"],
            "generation_events_before_unseal": phase["generation_events_before_first_unseal"],
            "pre_unseal_target_access_count": phase["pre_unseal_target_access_count"],
            "atomic_unseal_batches": phase["target_unseal_batches"],
            "target_records_unsealed": tournament["counts"]["target_records_unsealed"],
            "post_unseal_generation_count": phase["post_unseal_generation_count"],
            "post_unseal_tuning_events": phase["post_unseal_tuning_events"],
        },
        "representative_pass": {
            "world_id": pass_world["world_id"],
            "target_hypothesis": pass_world["unsealed_target"]["hypothesis"],
            "surviving_families": pass_world["pareto_eligible_families"],
            "candidate_hypotheses": _candidate_hypotheses(pass_world),
            "decision": pass_world["decision"],
        },
        "fixed_budget_reject": {
            "world_id": reject_world["world_id"],
            "target_hypothesis": reject_world["unsealed_target"]["hypothesis"],
            "candidate_hypotheses": _candidate_hypotheses(reject_world),
            "surviving_families": reject_world["pareto_eligible_families"],
            "decision": reject_world["decision"],
        },
        "robustness": {
            "registered_seeds": robustness["counts"]["registered_seeds"],
            "cpu_replay_passes": robustness["counts"]["cpu_replay_passes"],
            "candidate_evaluation_replays": robustness["counts"]["candidate_evaluation_replays"],
            "gate_outcome_comparisons": robustness["counts"]["gate_outcome_comparisons"],
            "pareto_recomputations": robustness["counts"]["pareto_recomputations"],
            "candidate_overlap_per_seed": {
                "intersection": 21,
                "union": 21,
                "jaccard": {"numerator": 1, "denominator": 1},
            },
            "gate_status_stable": all(
                row["gate_outcomes_stable"] for row in robustness["cpu_exact_replays"]
            ),
            "fronts_stable": all(row["fronts_stable"] for row in robustness["cpu_exact_replays"]),
        },
        "ablation": {
            "dependent_family": "bayesian",
            "pass_to_reject_worlds": bayesian_ablation["world_pass_to_reject_count"],
            "front_members_removed": bayesian_ablation["front_assignment_change_count"],
            "other_family_decision_changes": sum(
                row["world_pass_to_reject_count"] for row in other_ablations
            ),
            "other_family_front_changes": sum(
                row["front_assignment_change_count"] for row in other_ablations
            ),
        },
        "prior_art": {
            "corpus_counts": prior_art.counts,
            "candidate_queries": len(candidate_hashes),
            "status_counts": dict(sorted(corpus_statuses.items())),
            "interpretation": (
                "Every candidate is absent from this exact equation-universe snapshot; corpus "
                "absence is a bounded lookup result and never a novelty certificate."
            ),
        },
        "claims": dict(CLAIMS),
        "scope": (
            "human-readable projection of two validated campaign receipts and one static "
            "prior-art import; it changes no candidate, gate, proof, rank, or promotion decision"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_report(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "bindings",
        "chronology",
        "representative_pass",
        "fixed_budget_reject",
        "robustness",
        "ablation",
        "prior_art",
        "claims",
        "scope",
        "content_sha256",
    }
    if set(value) != expected or value.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("research report schema changed")
    if value.get("content_sha256") != canonical_sha256(
        {key: child for key, child in value.items() if key != "content_sha256"}
    ):
        raise ValueError("research report self-seal changed")
    if value.get("claims") != CLAIMS:
        raise ValueError("research report claim boundary changed")
    if (
        value["chronology"]["generation_events_before_unseal"] != 21
        or value["chronology"]["pre_unseal_target_access_count"] != 0
        or value["chronology"]["atomic_unseal_batches"] != 1
        or value["fixed_budget_reject"]["surviving_families"] != []
        or value["ablation"]
        != {
            "dependent_family": "bayesian",
            "pass_to_reject_worlds": 2,
            "front_members_removed": 2,
            "other_family_decision_changes": 0,
            "other_family_front_changes": 0,
        }
        or value["prior_art"]["status_counts"] != {"absent_from_this_corpus": 21}
    ):
        raise ValueError("research report semantic boundary changed")


def render_markdown(report: Mapping[str, Any]) -> str:
    validate_report(report)
    chronology = report["chronology"]
    success = report["representative_pass"]
    rejection = report["fixed_budget_reject"]
    robustness = report["robustness"]
    ablation = report["ablation"]
    prior = report["prior_art"]
    rejection_rows = ", ".join(
        f"`{family}={hypothesis}`"
        for family, hypothesis in rejection["candidate_hypotheses"].items()
    )
    bindings = report["bindings"]
    lines = [
        "# Prospective blind tournament: success, rejection, and dependence",
        "",
        "> This report is generated from validated receipts. A bounded PASS is not general",
        "> discovery, stable replay is not truth, and corpus absence is not novelty.",
        "",
        "## Receipt bindings",
        "",
        "| Evidence | Path | File SHA-256 | Content SHA-256 |",
        "| --- | --- | --- | --- |",
        f"| Tournament | `{bindings['tournament']['path']}` | `{bindings['tournament']['file_sha256']}` | `{bindings['tournament']['content_sha256']}` |",
        f"| Robustness and ablation | `{bindings['robustness']['path']}` | `{bindings['robustness']['file_sha256']}` | `{bindings['robustness']['content_sha256']}` |",
        f"| Prior-art seed | `{bindings['prior_art']['seed_path']}` | `{bindings['prior_art']['seed_file_sha256']}` | import `{bindings['prior_art']['import_content_sha256']}` |",
        f"| Prior-art policy | `{bindings['prior_art']['policy_path']}` | `{bindings['prior_art']['policy_file_sha256']}` | graph `{bindings['prior_art']['graph_content_sha256']}` |",
        f"| Prior-art audit | `{bindings['prior_art']['audit_path']}` | `{bindings['prior_art']['audit_file_sha256']}` | import validated |",
        "",
        "## Chronology: what was hidden, and when",
        "",
        f"The campaign preregistered **{chronology['preregistered_worlds']} worlds** and **{chronology['native_generator_families_per_world']} native generator families per world**. All **{chronology['generation_events_before_unseal']} generation events** completed before the single atomic unseal. The pre-unseal target-access count was **{chronology['pre_unseal_target_access_count']}**. That one batch disclosed **{chronology['target_records_unsealed']} target records**; post-unseal generation and tuning both remained zero.",
        "",
        "## A bounded PASS",
        "",
        f"In `{success['world_id']}`, the sealed target was hypothesis **{success['target_hypothesis']}**. The Bayesian proposal selected the same registered hypothesis and survived both hard gates. It was the only surviving family in this world. This proves only that one target-blind proposal hit one finite registered target under the fixed budget.",
        "",
        "## The honest fixed-budget REJECT",
        "",
        f"In `{rejection['world_id']}`, the target was hypothesis **{rejection['target_hypothesis']}**. The seven target-blind proposals were {rejection_rows}. None matched, so the world terminated as `reject_fixed_budget_exhausted_without_holdout_match`, with no metric receipt and no Pareto front. The REJECT is evidence about this search budget, not a proof that the target is undiscoverable.",
        "",
        "## Replay robustness",
        "",
        f"Across **{robustness['registered_seeds']}** preregistered order perturbations, all **{robustness['cpu_replay_passes']}** CPU replay passes were exact: **{robustness['candidate_evaluation_replays']}** candidate evaluations, **{robustness['gate_outcome_comparisons']}** gate comparisons, and **{robustness['pareto_recomputations']}** Pareto recomputations. Candidate overlap was 21/21 for every seed; gate statuses and fronts were stable. This is replay invariance of a sealed candidate set, not evidence that new search seeds discover the same mathematics.",
        "",
        "## Bayesian-only dependence",
        "",
        f"Removing Bayesian proposals changed **{ablation['pass_to_reject_worlds']}** worlds from PASS to REJECT and removed **{ablation['front_members_removed']}** front members. Removing any of the other six families caused **{ablation['other_family_decision_changes']}** decision changes and **{ablation['other_family_front_changes']}** front changes. The present successes are therefore Bayesian-dependent, not portfolio-robust.",
        "",
        "## Prior art: absence is not novelty",
        "",
        f"The validated static equation-universe import contains {prior['corpus_counts']['sources']} sources and {prior['corpus_counts']['equations']} equations. All **{prior['candidate_queries']}** tournament candidate hashes returned `absent_from_this_corpus`. This means only that those exact content identities are absent from that exact snapshot. It does not establish semantic novelty, historical priority, equivalence-class novelty, usefulness, or truth.",
        "",
        "## Sharp boundary",
        "",
        "The tournament provides two bounded PASS worlds, one honest REJECT world, exact replay, and a dependency diagnosis. It does not establish general discovery, scientific truth, novelty, or promotion eligibility. The next meaningful gate is an independently authored external world with a substantive proof oracle.",
        "",
        f"Report content SHA-256: `{report['content_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def render_ipynb(report: Mapping[str, Any]) -> dict[str, Any]:
    markdown = render_markdown(report)
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"report_content_sha256": report["content_sha256"]},
                "source": markdown.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "language_info": {"name": "markdown"},
            "report_schema": REPORT_SCHEMA,
            "report_content_sha256": report["content_sha256"],
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_outputs(root: Path) -> tuple[str, str]:
    report = build_report(root)
    validate_report(report)
    markdown = render_markdown(report)
    notebook = json.dumps(render_ipynb(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return markdown, notebook


def _write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"immutable research-report output differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    markdown, notebook = build_outputs(root)
    _write_immutable(_resolve(root, MARKDOWN_PATH), markdown)
    _write_immutable(_resolve(root, IPYNB_PATH), notebook)
    print(
        json.dumps(
            {
                "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                "ipynb_sha256": hashlib.sha256(notebook.encode()).hexdigest(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
