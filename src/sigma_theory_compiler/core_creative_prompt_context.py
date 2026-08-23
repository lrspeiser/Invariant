"""Bind verified first-principles context into every live core Claude prompt.

The external benchmark campaign remains a reproducible public control.  The core application wraps
its provider transport with this module so that validated symmetry/dimension briefs, learned
diagonal and state-pair invariants, typed grammar kinds, proof routes, and origin-label policy reach
Claude without changing the deterministic control receipt.  The wrapper records the exact context
seal beside the provider prompt hash.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .claude_creativity_api import ClaudeCreativityError, Transport, urllib_transport
from .external_claude_transport import ProviderCompatibleClaudeTransport
from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-core-creative-prompt-context-1.3"
FIRST_PRINCIPLES_BRIEF_COUNT = 5
LEARNED_INVARIANT_BRIEF_COUNT = 3
STATE_PAIR_INVARIANT_BRIEF_COUNT = 3
ORIGIN_ASSESSMENTS = [
    "cross_domain_synthesis",
    "known_rewrite",
    "proposed_new_construction",
    "uncertain",
]
TYPED_FORMULA_KINDS = [
    "finite_product",
    "finite_sum",
    "generating_function",
    "modular_relation",
    "recurrence",
    "tensor_identity",
    "variational_functional",
]
PROOF_MECHANISMS = [
    "induction",
    "invariant_preservation",
    "bijection_or_involution",
    "minimal_counterexample_descent",
    "transform_and_extract",
    "contradiction",
]


class CoreCreativePromptContextError(ValueError):
    """The core prompt context or its provider binding failed closed."""


def _strict_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CoreCreativePromptContextError(f"{label} keys changed")
    return value


def build_creative_prompt_context(
    symmetry_dimension: Mapping[str, Any],
    learned_invariants: Mapping[str, Any],
    state_pair_invariants: Mapping[str, Any],
    expanded_grammar: Mapping[str, Any],
    proof_plan_search: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the bounded, public creativity context used by the core live lane."""

    if expanded_grammar.get("summary", {}).get("admitted_formula_kinds") != TYPED_FORMULA_KINDS:
        raise CoreCreativePromptContextError("typed formula context changed")
    if proof_plan_search.get("summary", {}).get("mechanisms") != PROOF_MECHANISMS:
        raise CoreCreativePromptContextError("independent proof context changed")

    briefs = []
    for result in symmetry_dimension.get("results", []):
        creative_brief = _strict_keys(
            result.get("creative_brief"),
            {
                "candidate_invariant_coordinates",
                "constraint_statement",
                "llm_origin_assessment_labels",
                "novelty_caution",
            },
            "first-principles creative brief",
        )
        if creative_brief["llm_origin_assessment_labels"] != [
            "known_rewrite",
            "cross_domain_synthesis",
            "proposed_new_construction",
            "uncertain",
        ]:
            raise CoreCreativePromptContextError("creative brief origin labels changed")
        briefs.append(
            {
                "candidate_invariant_coordinates": list(
                    creative_brief["candidate_invariant_coordinates"]
                ),
                "constraint_statement": creative_brief["constraint_statement"],
                "domain": result.get("domain"),
                "forced_form": result.get("forced_form", {}).get("statement"),
                "invariant_coordinate_arity": result.get("forced_form", {}).get(
                    "free_function_arity"
                ),
                "novelty_caution": creative_brief["novelty_caution"],
                "problem_id": result.get("problem_id"),
            }
        )
    briefs.sort(key=lambda item: str(item["problem_id"]))
    if len(briefs) != FIRST_PRINCIPLES_BRIEF_COUNT or any(
        not isinstance(item["problem_id"], str)
        or not isinstance(item["domain"], str)
        or not isinstance(item["forced_form"], str)
        or not isinstance(item["candidate_invariant_coordinates"], list)
        or not item["candidate_invariant_coordinates"]
        or any(
            not isinstance(coordinate, str) or not coordinate
            for coordinate in item["candidate_invariant_coordinates"]
        )
        or len(set(item["candidate_invariant_coordinates"]))
        != len(item["candidate_invariant_coordinates"])
        or item["invariant_coordinate_arity"]
        != len(item["candidate_invariant_coordinates"])
        for item in briefs
    ) or not any(item["invariant_coordinate_arity"] > 1 for item in briefs):
        raise CoreCreativePromptContextError("first-principles brief coverage changed")

    if learned_invariants.get("summary", {}).get("status") != (
        "PASS_LEARNED_MULTI_INVARIANT_CONTROLS"
    ):
        raise CoreCreativePromptContextError("learned invariant context changed")
    learned_briefs = []
    for result in learned_invariants.get("results", []):
        creative_brief = _strict_keys(
            result.get("creative_brief"),
            {
                "candidate_invariant_coordinates",
                "constraint_statement",
                "deployment_repaired_coordinates",
                "identifiability_status",
                "llm_origin_assessment_labels",
                "novelty_caution",
            },
            "learned invariant creative brief",
        )
        if creative_brief["llm_origin_assessment_labels"] != [
            "known_rewrite",
            "cross_domain_synthesis",
            "proposed_new_construction",
            "uncertain",
        ]:
            raise CoreCreativePromptContextError("learned invariant origin labels changed")
        learned_briefs.append(
            {
                "candidate_invariant_coordinates": list(
                    creative_brief["candidate_invariant_coordinates"]
                ),
                "constraint_statement": creative_brief["constraint_statement"],
                "deployment_repaired_coordinates": list(
                    creative_brief["deployment_repaired_coordinates"]
                ),
                "domain": result.get("domain"),
                "identifiability_status": creative_brief["identifiability_status"],
                "novelty_caution": creative_brief["novelty_caution"],
                "problem_id": result.get("problem_id"),
            }
        )
    learned_briefs.sort(key=lambda item: str(item["problem_id"]))
    learned_statuses = {
        item["identifiability_status"] for item in learned_briefs
    }
    if (
        len(learned_briefs) != LEARNED_INVARIANT_BRIEF_COUNT
        or learned_statuses
        != {
            "PASS_LEARNED_INVARIANT_BASIS",
            "REJECT_TRAIN_ONLY_INVARIANT_SPACE",
            "UNDERDETERMINED_RETAIN_CANDIDATE_SUBSPACE",
        }
        or any(
            not isinstance(item["problem_id"], str)
            or not isinstance(item["domain"], str)
            or not item["candidate_invariant_coordinates"]
            or not item["deployment_repaired_coordinates"]
            for item in learned_briefs
        )
    ):
        raise CoreCreativePromptContextError("learned invariant brief coverage changed")

    if state_pair_invariants.get("summary", {}).get("status") != (
        "PASS_EXACT_MATRIX_AND_NONLINEAR_STATE_PAIR_CONTROLS"
    ):
        raise CoreCreativePromptContextError("state-pair invariant context changed")
    state_pair_briefs = []
    for result in state_pair_invariants.get("results", []):
        creative_brief = _strict_keys(
            result.get("creative_brief"),
            {
                "action_kind",
                "candidate_invariant_coordinates",
                "constraint_statement",
                "deployment_repaired_coordinates",
                "identifiability_status",
                "llm_origin_assessment_labels",
                "novelty_caution",
                "retained_linear_invariant_basis",
            },
            "state-pair invariant creative brief",
        )
        if creative_brief["llm_origin_assessment_labels"] != [
            "known_rewrite",
            "cross_domain_synthesis",
            "proposed_new_construction",
            "uncertain",
        ]:
            raise CoreCreativePromptContextError("state-pair invariant origin labels changed")
        state_pair_briefs.append(
            {
                "action_kind": creative_brief["action_kind"],
                "candidate_invariant_coordinates": list(
                    creative_brief["candidate_invariant_coordinates"]
                ),
                "constraint_statement": creative_brief["constraint_statement"],
                "deployment_repaired_coordinates": list(
                    creative_brief["deployment_repaired_coordinates"]
                ),
                "domain": result.get("domain"),
                "identifiability_status": creative_brief["identifiability_status"],
                "novelty_caution": creative_brief["novelty_caution"],
                "problem_id": result.get("problem_id"),
                "retained_linear_invariant_basis": list(
                    creative_brief["retained_linear_invariant_basis"]
                ),
            }
        )
    state_pair_briefs.sort(key=lambda item: str(item["problem_id"]))
    if (
        len(state_pair_briefs) != STATE_PAIR_INVARIANT_BRIEF_COUNT
        or {item["action_kind"] for item in state_pair_briefs}
        != {"matrix_conjugation", "matrix_orthogonal", "nonlinear_polynomial"}
        or any(
            not isinstance(item["problem_id"], str)
            or not isinstance(item["domain"], str)
            or item["identifiability_status"]
            != "PASS_EXACT_STATE_PAIR_INVARIANT_BASIS"
            or not item["candidate_invariant_coordinates"]
            or not item["deployment_repaired_coordinates"]
            or not item["retained_linear_invariant_basis"]
            for item in state_pair_briefs
        )
    ):
        raise CoreCreativePromptContextError("state-pair invariant brief coverage changed")

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "creativity_policy": {
            "creativity_is_primary": True,
            "generate_multiple_mechanisms_before_falsification": True,
            "learn_matrix_and_nonlinear_actions_from_state_pairs": True,
            "origin_labels_are_fallible_lineage_assessments": True,
            "retain_every_schema_admitted_idea": True,
            "retain_failed_and_underdetermined_invariant_branches": True,
            "uncertainty_does_not_prune": True,
        },
        "first_principles_briefs": briefs,
        "independent_proof_mechanisms": list(PROOF_MECHANISMS),
        "learned_invariant_briefs": learned_briefs,
        "origin_assessment_labels": list(ORIGIN_ASSESSMENTS),
        "state_pair_invariant_briefs": state_pair_briefs,
        "typed_formula_kinds": list(TYPED_FORMULA_KINDS),
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_creative_prompt_context(body)
    return body


def validate_creative_prompt_context(value: Mapping[str, Any]) -> None:
    _strict_keys(
        value,
        {
            "content_sha256",
            "creativity_policy",
            "first_principles_briefs",
            "independent_proof_mechanisms",
            "learned_invariant_briefs",
            "origin_assessment_labels",
            "schema_version",
            "state_pair_invariant_briefs",
            "typed_formula_kinds",
        },
        "core creative prompt context",
    )
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CoreCreativePromptContextError("core creative prompt context seal changed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CoreCreativePromptContextError("core creative prompt context schema changed")
    if value.get("origin_assessment_labels") != ORIGIN_ASSESSMENTS:
        raise CoreCreativePromptContextError("core origin assessment policy changed")
    if value.get("typed_formula_kinds") != TYPED_FORMULA_KINDS:
        raise CoreCreativePromptContextError("core typed formula coverage changed")
    if value.get("independent_proof_mechanisms") != PROOF_MECHANISMS:
        raise CoreCreativePromptContextError("core proof mechanism coverage changed")
    policy = _strict_keys(
        value.get("creativity_policy"),
        {
            "creativity_is_primary",
            "generate_multiple_mechanisms_before_falsification",
            "learn_matrix_and_nonlinear_actions_from_state_pairs",
            "origin_labels_are_fallible_lineage_assessments",
            "retain_every_schema_admitted_idea",
            "retain_failed_and_underdetermined_invariant_branches",
            "uncertainty_does_not_prune",
        },
        "core creativity policy",
    )
    if any(item is not True for item in policy.values()):
        raise CoreCreativePromptContextError("core creativity policy weakened")
    briefs = value.get("first_principles_briefs")
    if not isinstance(briefs, list) or len(briefs) != FIRST_PRINCIPLES_BRIEF_COUNT:
        raise CoreCreativePromptContextError("core first-principles brief coverage changed")
    has_multiple_coordinates = False
    for brief in briefs:
        _strict_keys(
            brief,
            {
                "candidate_invariant_coordinates",
                "constraint_statement",
                "domain",
                "forced_form",
                "invariant_coordinate_arity",
                "novelty_caution",
                "problem_id",
            },
            "core first-principles brief",
        )
        coordinates = brief["candidate_invariant_coordinates"]
        arity = brief["invariant_coordinate_arity"]
        if (
            not isinstance(coordinates, list)
            or not coordinates
            or any(not isinstance(item, str) or not item for item in coordinates)
            or len(set(coordinates)) != len(coordinates)
            or not isinstance(arity, int)
            or isinstance(arity, bool)
            or arity != len(coordinates)
        ):
            raise CoreCreativePromptContextError(
                "core first-principles coordinate basis changed"
            )
        has_multiple_coordinates = has_multiple_coordinates or arity > 1
    if not has_multiple_coordinates:
        raise CoreCreativePromptContextError(
            "core first-principles multi-coordinate coverage changed"
        )
    learned_briefs = value.get("learned_invariant_briefs")
    if (
        not isinstance(learned_briefs, list)
        or len(learned_briefs) != LEARNED_INVARIANT_BRIEF_COUNT
    ):
        raise CoreCreativePromptContextError("core learned invariant brief coverage changed")
    learned_statuses: set[str] = set()
    for brief in learned_briefs:
        _strict_keys(
            brief,
            {
                "candidate_invariant_coordinates",
                "constraint_statement",
                "deployment_repaired_coordinates",
                "domain",
                "identifiability_status",
                "novelty_caution",
                "problem_id",
            },
            "core learned invariant brief",
        )
        coordinates = brief["candidate_invariant_coordinates"]
        repaired = brief["deployment_repaired_coordinates"]
        status = brief["identifiability_status"]
        if (
            not isinstance(coordinates, list)
            or not coordinates
            or any(not isinstance(item, str) or not item for item in coordinates)
            or not isinstance(repaired, list)
            or not repaired
            or any(not isinstance(item, str) or not item for item in repaired)
            or not isinstance(status, str)
        ):
            raise CoreCreativePromptContextError("core learned invariant branch changed")
        learned_statuses.add(status)
    if learned_statuses != {
        "PASS_LEARNED_INVARIANT_BASIS",
        "REJECT_TRAIN_ONLY_INVARIANT_SPACE",
        "UNDERDETERMINED_RETAIN_CANDIDATE_SUBSPACE",
    }:
        raise CoreCreativePromptContextError("core learned invariant outcomes changed")
    state_pair_briefs = value.get("state_pair_invariant_briefs")
    if (
        not isinstance(state_pair_briefs, list)
        or len(state_pair_briefs) != STATE_PAIR_INVARIANT_BRIEF_COUNT
    ):
        raise CoreCreativePromptContextError("core state-pair invariant brief coverage changed")
    action_kinds: set[str] = set()
    for brief in state_pair_briefs:
        _strict_keys(
            brief,
            {
                "action_kind",
                "candidate_invariant_coordinates",
                "constraint_statement",
                "deployment_repaired_coordinates",
                "domain",
                "identifiability_status",
                "novelty_caution",
                "problem_id",
                "retained_linear_invariant_basis",
            },
            "core state-pair invariant brief",
        )
        coordinates = brief["candidate_invariant_coordinates"]
        repaired = brief["deployment_repaired_coordinates"]
        retained = brief["retained_linear_invariant_basis"]
        if (
            not isinstance(coordinates, list)
            or not coordinates
            or any(not isinstance(item, str) or not item for item in coordinates)
            or not isinstance(repaired, list)
            or not repaired
            or any(not isinstance(item, str) or not item for item in repaired)
            or not isinstance(retained, list)
            or not retained
            or any(not isinstance(item, str) or not item for item in retained)
            or brief["identifiability_status"]
            != "PASS_EXACT_STATE_PAIR_INVARIANT_BASIS"
            or not isinstance(brief["action_kind"], str)
        ):
            raise CoreCreativePromptContextError("core state-pair invariant branch changed")
        action_kinds.add(brief["action_kind"])
    if action_kinds != {"matrix_conjugation", "matrix_orthogonal", "nonlinear_polynomial"}:
        raise CoreCreativePromptContextError("core state-pair action coverage changed")
    if len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")) > 24_576:
        raise CoreCreativePromptContextError("core creative prompt context exceeds its byte budget")


class FirstPrinciplesContextTransport(ProviderCompatibleClaudeTransport):
    """Inject one validated context into each Messages request and bind its seal."""

    def __init__(
        self,
        creative_context: Mapping[str, Any],
        transport: Transport = urllib_transport,
    ) -> None:
        validate_creative_prompt_context(creative_context)
        super().__init__(transport)
        self.creative_context = json.loads(json.dumps(creative_context))
        self.creative_context_sha256 = str(creative_context["content_sha256"])

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        bound_body = body
        if method == "POST" and body is not None:
            try:
                request = json.loads(body)
                prompt = json.loads(request["messages"][0]["content"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ClaudeCreativityError("core Claude request envelope changed") from error
            if not isinstance(prompt, dict) or "creative_context" in prompt:
                raise ClaudeCreativityError("core Claude prompt context slot changed")
            prompt["creative_context"] = self.creative_context
            request["messages"][0]["content"] = json.dumps(
                prompt, sort_keys=True, separators=(",", ":")
            )
            bound_body = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()

        status, response = super().__call__(method, url, headers, bound_body, timeout)
        response_id = response.get("id")
        if method == "POST" and status == 200 and isinstance(response_id, str):
            evidence = dict(self._evidence.get(response_id, {}))
            evidence.update(
                {
                    "creative_context_injected": True,
                    "creative_context_sha256": self.creative_context_sha256,
                }
            )
            self._evidence[response_id] = evidence
        return status, response


__all__ = [
    "FIRST_PRINCIPLES_BRIEF_COUNT",
    "LEARNED_INVARIANT_BRIEF_COUNT",
    "SCHEMA_VERSION",
    "CoreCreativePromptContextError",
    "FirstPrinciplesContextTransport",
    "build_creative_prompt_context",
    "validate_creative_prompt_context",
]
