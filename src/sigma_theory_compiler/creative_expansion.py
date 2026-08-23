"""Curiosity-first expansion of retained ideas into proof plans and recombinations."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

from .idea_lineage import validate_idea_archive
from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-creative-expansion-1.0"


class CreativeExpansionError(ValueError):
    """The independent creative expansion or its non-pruning seal changed."""


_PROOF_TEMPLATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "strengthened_induction",
        (
            "choose an induction variable independently of the proposed proof plan",
            "strengthen the statement with a preserved invariant",
            "prove base and transition obligations",
        ),
    ),
    (
        "bijection_or_involution",
        (
            "identify two counted or weighted structures",
            "search for a weight-preserving bijection or sign-reversing involution",
            "prove inverse and boundary cases",
        ),
    ),
    (
        "minimal_counterexample_descent",
        (
            "assume a minimal counterexample in a declared well-order",
            "construct a strictly smaller counterexample or contradiction",
            "audit descent termination and exceptional strata",
        ),
    ),
    (
        "transform_and_extract",
        (
            "move to a generating, Fourier, Laplace, or z-transform representation",
            "derive a simpler algebraic or differential relation",
            "extract coefficients or invert the transform with domain obligations",
        ),
    ),
    (
        "contradiction_via_invariant",
        (
            "negate the target under explicit premises",
            "propagate a conserved, monotone, parity, or modular invariant",
            "derive an incompatible boundary or extremal condition",
        ),
    ),
    (
        "variational_or_dual_certificate",
        (
            "search for a functional, dual object, or certificate",
            "derive stationarity or complementary conditions",
            "replay boundary terms and equality conditions",
        ),
    ),
)


def _independent_proof_plans(idea: Mapping[str, Any]) -> list[dict[str, Any]]:
    plans = []
    for mechanism, steps in _PROOF_TEMPLATES:
        body = {
            "applicability_status": "UNTESTED_RETAIN_FOR_SEARCH",
            "idea_lineage_id": idea["lineage_id"],
            "independent_from_llm_declared_proof_plan": True,
            "mechanism": mechanism,
            "steps": list(steps),
        }
        plans.append(
            {
                **body,
                "plan_id": "plan." + canonical_sha256(body)[:24],
                "proof_mechanism_sha256": canonical_sha256(
                    {"mechanism": mechanism, "steps": list(steps)}
                ),
                "retention_status": "RETAINED_UNTIL_APPLICABILITY_TESTED",
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


def build_creative_expansion(archive: Mapping[str, Any]) -> dict[str, Any]:
    validate_idea_archive(archive)
    ideas = list(archive["ideas"])
    plans = [plan for idea in ideas for plan in _independent_proof_plans(idea)]
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
        "policy": {
            "creative_branches_are_claims": False,
            "failed_applicability_deletes_plan": False,
            "formula_and_proof_plan_spaces_searched_independently": True,
            "known_rewrites_are_recombination_seeds": True,
            "verification_creates_status_and_repair_edges_not_deletion": True,
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
            "recombination_branches_retained": len(recombinations),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_creative_expansion(body)
    return body


def validate_creative_expansion(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CreativeExpansionError("creative expansion content seal changed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CreativeExpansionError("creative expansion schema changed")
    policy = value.get("policy", {})
    plans = value.get("independent_proof_plans", [])
    summary = value.get("summary", {})
    novelty = value.get("novelty_axes", {})
    if (
        policy.get("formula_and_proof_plan_spaces_searched_independently") is not True
        or policy.get("known_rewrites_are_recombination_seeds") is not True
        or policy.get("verification_creates_status_and_repair_edges_not_deletion") is not True
        or not isinstance(plans, list)
        or not plans
        or summary.get("independent_plans_retained") != len(plans)
        or summary.get("ideas_expanded", 0) * len(_PROOF_TEMPLATES) != len(plans)
        or any(
            plan.get("retention_status") != "RETAINED_UNTIL_APPLICABILITY_TESTED"
            for plan in plans
        )
        or novelty.get("separate_axes") is not True
        or novelty.get("behavior_novelty_is_literature_novelty") is not False
        or novelty.get("proof_mechanism_novelty_is_literature_novelty") is not False
    ):
        raise CreativeExpansionError("curiosity-first expansion invariant changed")
