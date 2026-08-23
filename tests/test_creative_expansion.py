from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler.creative_expansion import (
    build_creative_expansion,
    validate_creative_expansion,
)
from sigma_theory_compiler.idea_lineage import build_idea_archive
from sigma_theory_compiler.independent_proof_plan_search import run_proof_plan_search

ROOT = Path(__file__).resolve().parents[1]


def _campaign() -> dict[str, object]:
    hypotheses = []
    for index, assessment in enumerate(("known_rewrite", "cross_domain_synthesis")):
        hypotheses.append(
            {
                "expression": f"F{index}(n)",
                "falsifiers": ["shift the initial data"],
                "family": f"family-{index}",
                "hypothesis_id": f"hypothesis-{index}",
                "invariants": [f"invariant-{index}"],
                "known_analogues": [f"analogue-{index}"],
                "llm_origin_assessment": assessment,
                "proof_plan": ["model supplied plan must not control independent search"],
                "rationale": "Keep and expand this branch.",
                "representation": "other_typed_relation" if index else "finite_sum",
                "source_idea_domains": (
                    ["continuous dynamics"] if index else ["combinatorics", "domain-0"]
                ),
                "synthesis_note": "A bounded lineage statement, not a novelty claim.",
            }
        )
    return {
        "claude": {
            "calls": [
                {
                    "benchmark_id": "blind.test",
                    "output": {
                        "hypotheses": hypotheses,
                        "quarantine": {"rejected_hypotheses": 0},
                        "steering_actions": [],
                    },
                    "role": "proposer",
                    "status": "completed",
                }
            ]
        }
    }


def test_creativity_expands_every_idea_without_pruning() -> None:
    archive = build_idea_archive(_campaign())
    proof_plan_library = run_proof_plan_search(ROOT)
    expansion = build_creative_expansion(archive, proof_plan_library)
    validate_creative_expansion(expansion, proof_plan_library)
    assert expansion["summary"]["ideas_expanded"] == 2
    assert expansion["summary"]["independent_plans_retained"] == 12
    assert expansion["summary"]["recombination_branches_retained"] == 1
    assert expansion["novelty_axes"]["distinct_behavior_signatures"] == 2
    assert expansion["novelty_axes"]["distinct_proof_mechanism_signatures"] == 6
    assert expansion["policy"]["known_rewrites_are_recombination_seeds"] is True
    assert expansion["policy"]["missing_applicability_features_delete_plan"] is False
    assert all(
        plan["retention_status"] == "RETAINED_FOR_CANDIDATE_PROOF_SEARCH"
        for plan in expansion["independent_proof_plans"]
    )
    assert {
        plan["applicability_status"] for plan in expansion["independent_proof_plans"]
    } == {"APPLICABLE_FEATURES_PRESENT", "REQUIRES_FEATURE_EVIDENCE_RETAINED"}
