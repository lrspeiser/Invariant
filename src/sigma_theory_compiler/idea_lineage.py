"""Non-pruning lineage archive for Claude mathematical suggestions.

The archive treats model origin labels as fallible self-assessments.  Exact checks and critics
may change an idea's claim readiness or create a repair branch, but they never delete the idea.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from .claude_creativity_api import CLAUDE_ORIGIN_ASSESSMENTS
from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-idea-lineage-archive-1.0"


class IdeaLineageError(ValueError):
    """A live suggestion could not be represented without pruning or ambiguity."""


_NEXT_ACTION = {
    "known_rewrite": "retain_as_rewrite_control_and_recombination_seed",
    "cross_domain_synthesis": "expand_cross_test_and_seek_mechanism",
    "proposed_new_construction": "expand_cross_test_and_run_prior_art_search",
    "uncertain": "retain_expand_and_request_more_lineage_evidence",
}


def _proposal_calls(campaign: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls = campaign.get("claude", {}).get("calls", [])
    if not isinstance(calls, list):
        raise IdeaLineageError("Claude calls are not an array")
    return [
        call
        for call in calls
        if isinstance(call, Mapping)
        and call.get("status") == "completed"
        and call.get("role") == "proposer"
    ]


def _critic_guidance(campaign: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calls = campaign.get("claude", {}).get("calls", [])
    for call in calls if isinstance(calls, list) else []:
        if (
            not isinstance(call, Mapping)
            or call.get("status") != "completed"
            or call.get("role") != "critic"
        ):
            continue
        output = call.get("output")
        actions = output.get("steering_actions", []) if isinstance(output, Mapping) else []
        for action in actions if isinstance(actions, list) else []:
            if not isinstance(action, Mapping):
                continue
            by_benchmark[str(call.get("benchmark_id"))].append(
                {
                    "blocker_kind": action.get("blocker_kind"),
                    "candidate_id": action.get("candidate_id"),
                    "distance_denominator": action.get("distance_denominator"),
                    "distance_numerator": action.get("distance_numerator"),
                    "repair_or_recombination": action.get("repair"),
                    "verdict": action.get("verdict"),
                }
            )
    return by_benchmark


def build_idea_archive(campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Retain every schema-admitted proposer suggestion and its provenance."""

    guidance = _critic_guidance(campaign)
    ideas: list[dict[str, Any]] = []
    quarantine_count = 0
    for call in _proposal_calls(campaign):
        output = call.get("output")
        if not isinstance(output, Mapping):
            raise IdeaLineageError("completed proposer call has no structured output")
        hypotheses = output.get("hypotheses")
        if not isinstance(hypotheses, list):
            raise IdeaLineageError("proposer hypotheses are not an array")
        quarantine = output.get("quarantine", {})
        if isinstance(quarantine, Mapping):
            quarantine_count += int(quarantine.get("rejected_hypotheses", 0))
        benchmark_id = str(call.get("benchmark_id"))
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, Mapping):
                raise IdeaLineageError("proposer hypothesis is not an object")
            assessment = hypothesis.get("llm_origin_assessment")
            if assessment not in CLAUDE_ORIGIN_ASSESSMENTS:
                raise IdeaLineageError("proposer hypothesis lacks an admitted origin assessment")
            proposal = {
                "benchmark_id": benchmark_id,
                "expression": hypothesis.get("expression"),
                "family": hypothesis.get("family"),
                "falsifiers": hypothesis.get("falsifiers"),
                "hypothesis_id": hypothesis.get("hypothesis_id"),
                "invariants": hypothesis.get("invariants"),
                "known_analogues": hypothesis.get("known_analogues"),
                "llm_origin_assessment": assessment,
                "proof_plan": hypothesis.get("proof_plan"),
                "rationale": hypothesis.get("rationale"),
                "representation": hypothesis.get("representation"),
                "source_idea_domains": hypothesis.get("source_idea_domains"),
                "synthesis_note": hypothesis.get("synthesis_note"),
            }
            ideas.append(
                {
                    **proposal,
                    "critic_guidance_for_benchmark": guidance.get(benchmark_id, []),
                    "idea_content_sha256": canonical_sha256(proposal),
                    "lineage_id": "idea." + canonical_sha256(proposal)[:24],
                    "next_action": _NEXT_ACTION[str(assessment)],
                    "retention_status": "RETAINED_ACTIVE",
                }
            )
    if not ideas:
        raise IdeaLineageError("live core run produced no retainable proposer ideas")
    counts = Counter(str(idea["llm_origin_assessment"]) for idea in ideas)
    archive: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "creativity_priority": 1,
            "critic_rejection_deletes_idea": False,
            "deduplication_deletes_lineage": False,
            "llm_origin_assessment_is_novelty_authority": False,
            "unsupported_execution_syntax_deletes_idea": False,
            "verification_failure_deletes_idea": False,
            "verification_role": "controls_claim_readiness_and_creates_repair_branches",
        },
        "ideas": ideas,
        "summary": {
            "admitted_origin_assessments": sorted(CLAUDE_ORIGIN_ASSESSMENTS),
            "ideas_received": len(ideas),
            "ideas_retained": len(ideas),
            "llm_origin_assessment_counts": dict(sorted(counts.items())),
            "schema_quarantined_hypotheses": quarantine_count,
        },
    }
    archive["content_sha256"] = canonical_sha256(archive)
    validate_idea_archive(archive)
    return archive


def validate_idea_archive(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise IdeaLineageError("idea archive content seal changed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise IdeaLineageError("idea archive schema changed")
    policy = value.get("policy", {})
    if (
        policy.get("creativity_priority") != 1
        or policy.get("llm_origin_assessment_is_novelty_authority") is not False
        or policy.get("verification_failure_deletes_idea") is not False
        or policy.get("unsupported_execution_syntax_deletes_idea") is not False
        or policy.get("critic_rejection_deletes_idea") is not False
    ):
        raise IdeaLineageError("non-pruning creativity policy changed")
    ideas = value.get("ideas", [])
    summary = value.get("summary", {})
    if (
        not isinstance(ideas, list)
        or not ideas
        or summary.get("ideas_received") != len(ideas)
        or summary.get("ideas_retained") != len(ideas)
        or any(idea.get("retention_status") != "RETAINED_ACTIVE" for idea in ideas)
        or any(
            idea.get("llm_origin_assessment") not in CLAUDE_ORIGIN_ASSESSMENTS
            for idea in ideas
        )
    ):
        raise IdeaLineageError("idea retention invariant changed")
