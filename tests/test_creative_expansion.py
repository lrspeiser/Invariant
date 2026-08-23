from __future__ import annotations

from sigma_theory_compiler.creative_expansion import (
    build_creative_expansion,
    validate_creative_expansion,
)
from sigma_theory_compiler.idea_lineage import build_idea_archive


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
                "representation": "linear_recurrence" if index else "finite_sum",
                "source_idea_domains": ["combinatorics", f"domain-{index}"],
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
    expansion = build_creative_expansion(archive)
    validate_creative_expansion(expansion)
    assert expansion["summary"]["ideas_expanded"] == 2
    assert expansion["summary"]["independent_plans_retained"] == 12
    assert expansion["summary"]["recombination_branches_retained"] == 1
    assert expansion["novelty_axes"]["distinct_behavior_signatures"] == 2
    assert expansion["novelty_axes"]["distinct_proof_mechanism_signatures"] == 6
    assert expansion["policy"]["known_rewrites_are_recombination_seeds"] is True
