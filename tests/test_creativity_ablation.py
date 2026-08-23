from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler.creativity_ablation import load_protocol, score_experiment

ROOT = Path(__file__).resolve().parents[1]


def _idea(index: int, useful: bool) -> dict[str, object]:
    score = 4 if useful else 1
    return {
        "behavior_sha256": f"{index:064x}",
        "human_reviews": [
            {
                "coherence": score,
                "followup_value": score,
                "nontriviality": score,
                "reviewer_id": "reviewer.alpha",
            },
            {
                "coherence": score,
                "followup_value": score,
                "nontriviality": score,
                "reviewer_id": "reviewer.beta",
            },
        ],
        "initial_check_status": "failed" if index % 2 else "untested",
        "later_used_as_parent": index % 2 == 1,
        "llm_origin_assessment": "uncertain",
        "prior_art_classification": "review_pending",
        "proof_mechanism_sha256": f"{index + 1000:064x}",
        "representation": "finite_sum" if index % 2 else "linear_recurrence",
        "source_domains": ["combinatorics", "dynamics"],
    }


def test_paired_blinded_protocol_can_establish_a_bounded_creativity_effect() -> None:
    protocol = load_protocol(ROOT)
    records = []
    for task in range(24):
        budget = {
            "calls": 2,
            "grammar_depth": 8,
            "tokens": 10000,
            "verifier_invocations": 5,
            "wall_clock_milliseconds": 60000,
        }
        records.extend(
            [
                {
                    "arm": "baseline",
                    "blinded_output_id": f"blind-a-{task}",
                    "ideas": [_idea(task * 10, True)],
                    "resource_budget": budget,
                    "task_id": f"task-{task}",
                    "tokens_used": 10000,
                    "typed_usable_ideas": 1,
                },
                {
                    "arm": "full_creativity_first",
                    "blinded_output_id": f"blind-b-{task}",
                    "ideas": [
                        _idea(task * 10 + 1, True),
                        _idea(task * 10 + 2, True),
                    ],
                    "resource_budget": budget,
                    "task_id": f"task-{task}",
                    "tokens_used": 10000,
                    "typed_usable_ideas": 2,
                },
            ]
        )
    result = score_experiment(
        {
            "baseline_commit": protocol["baseline_commit"],
            "experiment_id": protocol["experiment_id"],
            "records": records,
            "schema_version": "invariant-creativity-ablation-observations-1.0",
            "treatment_commit": "b" * 40,
        },
        protocol,
    )
    assert result["verdict"] == "MORE_CREATIVE_ON_PREREGISTERED_BOUNDED_PROTOCOL"
    assert result["paired_tasks"] == 24
    assert result["claims"]["literature_novelty_established"] is False
