"""Bind verified first-principles context into every live core Claude prompt.

The external benchmark campaign remains a reproducible public control.  The core application wraps
its provider transport with this module so that validated symmetry/dimension briefs, learned
diagonal and state-pair invariants, structured uncertainty branches, typed grammar kinds, proof
routes, and origin-label policy reach Claude without changing the deterministic control receipt.
The wrapper records the exact context seal beside the provider prompt hash.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .claude_creativity_api import ClaudeCreativityError, Transport, urllib_transport
from .external_claude_transport import ProviderCompatibleClaudeTransport
from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-core-creative-prompt-context-1.7"
FIRST_PRINCIPLES_BRIEF_COUNT = 5
LEARNED_INVARIANT_BRIEF_COUNT = 3
STATE_PAIR_INVARIANT_BRIEF_COUNT = 6
UNCERTAIN_INVARIANT_BRIEF_COUNT = 5
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
    "piecewise_relation",
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
    uncertain_invariants: Mapping[str, Any],
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
        "PASS_EXACT_TYPED_STATE_PAIR_INVARIANT_CONTROLS"
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
                "feature_grammar",
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
                "feature_grammar": dict(creative_brief["feature_grammar"]),
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
        != {
            "matrix_conjugation",
            "matrix_orthogonal",
            "nonlinear_polynomial",
            "nonlinear_polynomial_degree3",
            "rational_laurent",
            "transcendental_logarithmic",
        }
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

    if uncertain_invariants.get("summary", {}).get("status") != (
        "PASS_COUPLED_UNCERTAIN_INVARIANT_BRANCH_CONTROLS"
    ):
        raise CoreCreativePromptContextError("uncertain invariant context changed")
    uncertain_briefs = []
    for result in uncertain_invariants.get("results", []):
        creative_brief = _strict_keys(
            result.get("creative_brief"),
            {
                "candidate_invariant_coordinates",
                "constraint_statement",
                "dependence_semantics",
                "deployment_surviving_coordinates",
                "identifiability_status",
                "llm_origin_assessment_labels",
                "novelty_caution",
                "observation_mode",
                "retained_evidence_branches",
            },
            "uncertain invariant creative brief",
        )
        if creative_brief["llm_origin_assessment_labels"] != [
            "known_rewrite",
            "cross_domain_synthesis",
            "proposed_new_construction",
            "uncertain",
        ]:
            raise CoreCreativePromptContextError("uncertain invariant origin labels changed")
        uncertain_briefs.append(
            {
                "candidate_invariant_coordinates": list(
                    creative_brief["candidate_invariant_coordinates"]
                ),
                "constraint_statement": creative_brief["constraint_statement"],
                "dependence_semantics": creative_brief["dependence_semantics"],
                "deployment_surviving_coordinates": list(
                    creative_brief["deployment_surviving_coordinates"]
                ),
                "domain": result.get("domain"),
                "identifiability_status": creative_brief["identifiability_status"],
                "novelty_caution": creative_brief["novelty_caution"],
                "observation_mode": creative_brief["observation_mode"],
                "problem_id": result.get("problem_id"),
                "retained_evidence_branches": list(
                    creative_brief["retained_evidence_branches"]
                ),
            }
        )
    uncertain_briefs.sort(key=lambda item: str(item["problem_id"]))
    if (
        len(uncertain_briefs) != UNCERTAIN_INVARIANT_BRIEF_COUNT
        or {item["observation_mode"] for item in uncertain_briefs}
        != {
            "joint_support",
            "missingness",
            "noisy_interval",
            "one_sided_censoring",
            "unit_hypotheses",
        }
        or {item["identifiability_status"] for item in uncertain_briefs}
        != {
            "CENSORED_RETAIN_SET_VALUED_CANDIDATES",
            "DEPENDENT_RETAIN_JOINT_SUPPORT_COMPATIBLE_SET",
            "MISSINGNESS_RETAIN_PARTIALLY_OBSERVED_SET",
            "NOISY_RETAIN_INTERVAL_COMPATIBLE_SET",
            "UNIT_UNCERTAINTY_RETAIN_GLOBAL_HYPOTHESIS_BRANCHES",
        }
        or any(
            not isinstance(item["problem_id"], str)
            or not isinstance(item["domain"], str)
            or not item["candidate_invariant_coordinates"]
            or not item["deployment_surviving_coordinates"]
            for item in uncertain_briefs
        )
    ):
        raise CoreCreativePromptContextError("uncertain invariant brief coverage changed")

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "creativity_policy": {
            "creativity_is_primary": True,
            "generate_multiple_mechanisms_before_falsification": True,
            "learn_higher_degree_rational_and_logarithmic_features": True,
            "learn_matrix_and_nonlinear_actions_from_state_pairs": True,
            "origin_labels_are_fallible_lineage_assessments": True,
            "preserve_joint_and_unit_hypothesis_branches": True,
            "retain_every_schema_admitted_idea": True,
            "retain_failed_and_underdetermined_invariant_branches": True,
            "retain_set_valued_uncertainty_branches": True,
            "uncertainty_does_not_prune": True,
        },
        "first_principles_briefs": briefs,
        "independent_proof_mechanisms": list(PROOF_MECHANISMS),
        "learned_invariant_briefs": learned_briefs,
        "origin_assessment_labels": list(ORIGIN_ASSESSMENTS),
        "state_pair_invariant_briefs": state_pair_briefs,
        "typed_formula_kinds": list(TYPED_FORMULA_KINDS),
        "uncertain_invariant_briefs": uncertain_briefs,
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
            "uncertain_invariant_briefs",
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
            "learn_higher_degree_rational_and_logarithmic_features",
            "learn_matrix_and_nonlinear_actions_from_state_pairs",
            "origin_labels_are_fallible_lineage_assessments",
            "preserve_joint_and_unit_hypothesis_branches",
            "retain_every_schema_admitted_idea",
            "retain_failed_and_underdetermined_invariant_branches",
            "retain_set_valued_uncertainty_branches",
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
    feature_kinds: set[str] = set()
    for brief in state_pair_briefs:
        _strict_keys(
            brief,
            {
                "action_kind",
                "candidate_invariant_coordinates",
                "constraint_statement",
                "deployment_repaired_coordinates",
                "domain",
                "feature_grammar",
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
        feature_grammar = brief["feature_grammar"]
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
            or not isinstance(feature_grammar, Mapping)
            or feature_grammar.get("kind")
            not in {
                "laurent_monomials",
                "logarithmic_coordinates",
                "polynomial_monomials",
            }
        ):
            raise CoreCreativePromptContextError("core state-pair invariant branch changed")
        action_kinds.add(brief["action_kind"])
        feature_kinds.add(feature_grammar["kind"])
    if action_kinds != {
        "matrix_conjugation",
        "matrix_orthogonal",
        "nonlinear_polynomial",
        "nonlinear_polynomial_degree3",
        "rational_laurent",
        "transcendental_logarithmic",
    } or feature_kinds != {
        "laurent_monomials",
        "logarithmic_coordinates",
        "polynomial_monomials",
    }:
        raise CoreCreativePromptContextError("core state-pair action coverage changed")
    uncertain_briefs = value.get("uncertain_invariant_briefs")
    if (
        not isinstance(uncertain_briefs, list)
        or len(uncertain_briefs) != UNCERTAIN_INVARIANT_BRIEF_COUNT
    ):
        raise CoreCreativePromptContextError("core uncertain invariant brief coverage changed")
    modes: set[str] = set()
    statuses: set[str] = set()
    for brief in uncertain_briefs:
        _strict_keys(
            brief,
            {
                "candidate_invariant_coordinates",
                "constraint_statement",
                "dependence_semantics",
                "deployment_surviving_coordinates",
                "domain",
                "identifiability_status",
                "novelty_caution",
                "observation_mode",
                "problem_id",
                "retained_evidence_branches",
            },
            "core uncertain invariant brief",
        )
        candidates = brief["candidate_invariant_coordinates"]
        survivors = brief["deployment_surviving_coordinates"]
        evidence_branches = brief["retained_evidence_branches"]
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(not isinstance(item, str) or not item for item in candidates)
            or not isinstance(survivors, list)
            or not survivors
            or any(not isinstance(item, str) or not item for item in survivors)
            or not isinstance(brief["observation_mode"], str)
            or not isinstance(brief["identifiability_status"], str)
            or not isinstance(brief["dependence_semantics"], str)
            or brief["dependence_semantics"]
            != {
                "joint_support": "finite_joint_support_without_marginal_factorization",
                "missingness": "independent_marginal_bounds",
                "noisy_interval": "independent_marginal_bounds",
                "one_sided_censoring": "independent_marginal_bounds",
                "unit_hypotheses": "one_global_unit_hypothesis_per_candidate_branch",
            }.get(brief["observation_mode"])
            or not isinstance(evidence_branches, list)
            or len(evidence_branches) != len(candidates)
            or any(
                not isinstance(branch, Mapping)
                or set(branch) != {"branch_ids", "expression"}
                or branch["expression"] not in candidates
                or not isinstance(branch["branch_ids"], list)
                or not branch["branch_ids"]
                or any(
                    not isinstance(branch_id, str) or not branch_id
                    for branch_id in branch["branch_ids"]
                )
                for branch in evidence_branches
            )
            or len(candidates)
            != {
                "joint_support": 1,
                "missingness": 1,
                "noisy_interval": 3,
                "one_sided_censoring": 2,
                "unit_hypotheses": 2,
            }.get(brief["observation_mode"])
            or len(survivors) != 1
            or (
                brief["observation_mode"] == "joint_support"
                and evidence_branches
                != [{"branch_ids": ["finite_joint_support"], "expression": "a*b/c"}]
            )
            or (
                brief["observation_mode"] == "unit_hypotheses"
                and evidence_branches
                != [
                    {"branch_ids": ["b_three_c_two"], "expression": "a*b/c"},
                    {"branch_ids": ["b_half"], "expression": "a**2/b"},
                ]
            )
        ):
            raise CoreCreativePromptContextError("core uncertain invariant branch changed")
        modes.add(brief["observation_mode"])
        statuses.add(brief["identifiability_status"])
    if modes != {
        "joint_support",
        "missingness",
        "noisy_interval",
        "one_sided_censoring",
        "unit_hypotheses",
    } or statuses != {
        "CENSORED_RETAIN_SET_VALUED_CANDIDATES",
        "DEPENDENT_RETAIN_JOINT_SUPPORT_COMPATIBLE_SET",
        "MISSINGNESS_RETAIN_PARTIALLY_OBSERVED_SET",
        "NOISY_RETAIN_INTERVAL_COMPATIBLE_SET",
        "UNIT_UNCERTAINTY_RETAIN_GLOBAL_HYPOTHESIS_BRANCHES",
    }:
        raise CoreCreativePromptContextError("core uncertain invariant outcomes changed")
    if len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")) > 32_768:
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
