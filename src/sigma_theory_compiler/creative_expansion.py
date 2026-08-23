"""Curiosity-first expansion of retained ideas into proof plans and recombinations."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

from .idea_lineage import validate_idea_archive
from .independent_proof_plan_search import (
    APPLICABILITY_STATUSES,
    infer_candidate_capabilities,
    plan_templates,
    validate_proof_plan_search,
)
from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-creative-expansion-2.0"


class CreativeExpansionError(ValueError):
    """The independent creative expansion or its non-pruning seal changed."""


def _independent_proof_plans(
    idea: Mapping[str, Any], templates: tuple[Mapping[str, Any], ...]
) -> list[dict[str, Any]]:
    observed = set(infer_candidate_capabilities(idea))
    plans = []
    for template in templates:
        required = set(template["required_candidate_capabilities"])
        missing = sorted(required - observed)
        applicability = (
            "APPLICABLE_FEATURES_PRESENT"
            if not missing
            else "REQUIRES_FEATURE_EVIDENCE_RETAINED"
        )
        body = {
            "applicability_status": applicability,
            "idea_lineage_id": idea["lineage_id"],
            "independent_from_llm_declared_proof_plan": True,
            "mechanism": template["mechanism"],
            "missing_candidate_capabilities": missing,
            "observed_candidate_capabilities": sorted(observed),
            "rank": template["rank"],
            "required_candidate_capabilities": sorted(required),
            "route_id": template["route_id"],
            "search_closed_on_abstract_route": True,
            "steps": list(template["steps"]),
            "template_id": template["template_id"],
        }
        plans.append(
            {
                **body,
                "plan_id": "plan." + canonical_sha256(body)[:24],
                "proof_mechanism_sha256": template["proof_mechanism_sha256"],
                "retention_status": "RETAINED_FOR_CANDIDATE_PROOF_SEARCH",
            }
        )
    return plans


def _recombinations(ideas: list[Mapping[str, Any]], maximum: int = 64) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(ideas, 2):
        left_domains = set(left.get("source_idea_domains", []))
        right_domains = set(right.get("source_idea_domains", []))
        if (
            left.get("family") == right.get("family")
            and left_domains == right_domains
            and left.get("representation") == right.get("representation")
        ):
            continue
        body = {
            "construction": (
                "transport the invariants and proof mechanisms of the left idea into the "
                "representation and source domains of the right idea, then test both directions"
            ),
            "left_lineage_id": left["lineage_id"],
            "proposed_representation_pair": [
                left.get("representation"),
                right.get("representation"),
            ],
            "right_lineage_id": right["lineage_id"],
            "source_domains": sorted(left_domains | right_domains),
        }
        rows.append(
            {
                **body,
                "claim_readiness": "UNVERIFIED_CREATIVE_BRANCH",
                "recombination_id": "recombination." + canonical_sha256(body)[:24],
                "retention_status": "RETAINED_ACTIVE",
            }
        )
        if len(rows) >= maximum:
            break
    return rows


def build_creative_expansion(
    archive: Mapping[str, Any], proof_plan_library: Mapping[str, Any]
) -> dict[str, Any]:
    validate_idea_archive(archive)
    validate_proof_plan_search(proof_plan_library)
    ideas = list(archive["ideas"])
    templates = plan_templates(proof_plan_library)
    plans = [plan for idea in ideas for plan in _independent_proof_plans(idea, templates)]
    recombinations = _recombinations(ideas)
    behavior_hashes = {
        canonical_sha256(
            {
                "expression": idea.get("expression"),
                "invariants": idea.get("invariants"),
                "representation": idea.get("representation"),
            }
        )
        for idea in ideas
    }
    mechanism_hashes = {plan["proof_mechanism_sha256"] for plan in plans}
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_idea_archive_sha256": archive["content_sha256"],
        "proof_plan_library_content_sha256": proof_plan_library["content_sha256"],
        "policy": {
            "creative_branches_are_claims": False,
            "failed_applicability_deletes_plan": False,
            "formula_and_proof_plan_spaces_searched_independently": True,
            "known_rewrites_are_recombination_seeds": True,
            "verification_creates_status_and_repair_edges_not_deletion": True,
            "missing_applicability_features_delete_plan": False,
        },
        "independent_proof_plans": plans,
        "recombinations": recombinations,
        "novelty_axes": {
            "behavior_novelty_is_literature_novelty": False,
            "distinct_behavior_signatures": len(behavior_hashes),
            "distinct_proof_mechanism_signatures": len(mechanism_hashes),
            "proof_mechanism_novelty_is_literature_novelty": False,
            "separate_axes": True,
        },
        "summary": {
            "ideas_expanded": len(ideas),
            "independent_plans_retained": len(plans),
            "plan_templates_searched": len(templates),
            "recombination_branches_retained": len(recombinations),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_creative_expansion(body, proof_plan_library)
    return body


def validate_creative_expansion(
    value: Mapping[str, Any], proof_plan_library: Mapping[str, Any] | None = None
) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CreativeExpansionError("creative expansion content seal changed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CreativeExpansionError("creative expansion schema changed")
    policy = value.get("policy", {})
    plans = value.get("independent_proof_plans", [])
    summary = value.get("summary", {})
    novelty = value.get("novelty_axes", {})
    template_count = summary.get("plan_templates_searched", 0)
    if (
        policy.get("formula_and_proof_plan_spaces_searched_independently") is not True
        or policy.get("known_rewrites_are_recombination_seeds") is not True
        or policy.get("verification_creates_status_and_repair_edges_not_deletion") is not True
        or policy.get("missing_applicability_features_delete_plan") is not False
        or not isinstance(plans, list)
        or not plans
        or summary.get("independent_plans_retained") != len(plans)
        or template_count < 6
        or summary.get("ideas_expanded", 0) * template_count != len(plans)
        or any(
            plan.get("retention_status") != "RETAINED_FOR_CANDIDATE_PROOF_SEARCH"
            for plan in plans
        )
        or any(plan.get("applicability_status") not in APPLICABILITY_STATUSES for plan in plans)
        or any(plan.get("independent_from_llm_declared_proof_plan") is not True for plan in plans)
        or any(plan.get("search_closed_on_abstract_route") is not True for plan in plans)
        or any(
            set(plan.get("missing_candidate_capabilities", []))
            != set(plan.get("required_candidate_capabilities", []))
            - set(plan.get("observed_candidate_capabilities", []))
            for plan in plans
        )
        or novelty.get("separate_axes") is not True
        or novelty.get("behavior_novelty_is_literature_novelty") is not False
        or novelty.get("proof_mechanism_novelty_is_literature_novelty") is not False
    ):
        raise CreativeExpansionError("curiosity-first expansion invariant changed")
    if proof_plan_library is not None:
        validate_proof_plan_search(proof_plan_library)
        templates = plan_templates(proof_plan_library)
        if (
            value.get("proof_plan_library_content_sha256")
            != proof_plan_library.get("content_sha256")
            or template_count != len(templates)
            or {plan.get("template_id") for plan in plans}
            != {template.get("template_id") for template in templates}
            or {plan.get("proof_mechanism_sha256") for plan in plans}
            != {template.get("proof_mechanism_sha256") for template in templates}
        ):
            raise CreativeExpansionError("independent proof-plan library binding changed")
