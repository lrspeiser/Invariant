"""Live, fail-closed core application for creative formula and proof discovery.

Unlike the deterministic public Formula Discovery Job, this application requires authenticated
Claude participation.  It performs the full blind proposer/critic campaign in memory, persists only
credential-free evidence, and binds deterministic discovery plus multi-host verification receipts.
Claude is available across creative roles, never as a verifier or novelty authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .claim_specific_prior_art_portfolio import (
    validate_batch_receipt as validate_prior_art_portfolio_receipt,
)
from .claim_specific_prior_art_portfolio import (
    validate_preflight as validate_prior_art_portfolio_preflight,
)
from .claude_creativity_api import ClaudeRole
from .core_creative_prompt_context import (
    FirstPrinciplesContextTransport,
    build_creative_prompt_context,
    validate_creative_prompt_context,
)
from .core_credential import CredentialActivationError, activated_credential
from .creative_expansion import build_creative_expansion, validate_creative_expansion
from .creativity_component_knockout_preflight import (
    validate_receipt as validate_component_knockout_preflight,
)
from .dataset_challenge_suite import validate_dataset_challenges
from .declarative_discovery_operational_campaign import (
    validate_receipt as validate_operational_receipt,
)
from .expanded_math_grammar_controls import (
    validate_receipt as validate_expanded_grammar_receipt,
)
from .external_creativity_live_evidence import (
    build_evidence_from_receipt,
    validate_evidence,
)
from .external_creativity_multi_host import validate_receipt as validate_multi_host_receipt
from .external_creativity_validation import run_campaign
from .external_dataset_challenges import (
    validate_external_dataset_challenges,
)
from .external_structured_benchmarks import (
    RECEIPT_SCHEMA as EXTERNAL_STRUCTURED_RECEIPT_SCHEMA,
)
from .external_structured_benchmarks import load_stored_pack
from .idea_lineage import build_idea_archive, validate_idea_archive
from .independent_proof_plan_search import validate_proof_plan_search
from .learned_invariant_discovery import validate_receipt as validate_learned_invariants
from .level5_success_admission import validate_receipt as validate_level5_success_admission
from .retained_piecewise_descendant_campaign import (
    validate_receipt as validate_piecewise_descendant_campaign,
)
from .retained_piecewise_replay import validate_receipt as validate_piecewise_replay
from .serious_claim_verification_ladder import (
    REQUIRED_STAGES as SERIOUS_CLAIM_STAGES,
)
from .serious_claim_verification_ladder import (
    validate_receipt as validate_serious_claim_ladder,
)
from .sigma_core import canonical_sha256
from .state_pair_invariant_discovery import (
    validate_receipt as validate_state_pair_invariants,
)
from .symmetry_dimension_derivation import (
    validate_receipt as validate_symmetry_dimension_derivation,
)
from .uncertain_invariant_discovery import (
    validate_receipt as validate_uncertain_invariants,
)

CONFIG_PATH = "configs/core_creative_discovery.json"
OUTPUT_PATH = "runs/math/core-creative-discovery/live-runtime.json"
FAILED_CAMPAIGN_PATH = "work/core-creative-discovery/failed-live-campaign.json"
LIVE_CALL_JOURNAL_PATH = "work/core-creative-discovery/live-call-attempts.jsonl"
SOURCE_PATH = "src/sigma_theory_compiler/core_creative_discovery.py"
CLAUDE_API_SOURCE_PATH = "src/sigma_theory_compiler/claude_creativity_api.py"
EXTERNAL_CAMPAIGN_SOURCE_PATH = "src/sigma_theory_compiler/external_creativity_validation.py"
PROMPT_CONTEXT_SOURCE_PATH = "src/sigma_theory_compiler/core_creative_prompt_context.py"
SCHEMA_VERSION = "invariant-core-creative-discovery-runtime-3.1"
CONFIG_SCHEMA = "invariant-core-creative-discovery-config-3.1"


class CoreCreativeDiscoveryError(ValueError):
    """The live core application or one of its required gates failed closed."""


CampaignRunner = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _serialized_sha256(value: Mapping[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_context_runtime_evidence(
    creative_context: Mapping[str, Any], live_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe whether the preserved authenticated calls carried the current context."""

    validate_creative_prompt_context(creative_context)
    calls = live_evidence.get("calls", [])
    bound_calls = [
        call
        for call in calls
        if call.get("creative_context_injected") is True
        and call.get("creative_context_sha256") == creative_context["content_sha256"]
    ]
    context_claims = [
        call
        for call in calls
        if call.get("creative_context_injected") is True
        or call.get("creative_context_sha256") is not None
    ]
    if context_claims and len(bound_calls) != len(calls):
        raise CoreCreativeDiscoveryError("authenticated Claude prompt context is mixed or stale")
    context_bound = bool(calls) and len(bound_calls) == len(calls)
    return {
        "authenticated_calls_bound_to_context": context_bound,
        "bound_authenticated_calls": len(bound_calls),
        "content_sha256": creative_context["content_sha256"],
        "first_principles_briefs": len(creative_context["first_principles_briefs"]),
        "independent_proof_mechanisms": len(creative_context["independent_proof_mechanisms"]),
        "origin_assessment_labels": creative_context["origin_assessment_labels"],
        "learned_invariant_briefs": len(creative_context["learned_invariant_briefs"]),
        "state_pair_invariant_briefs": len(creative_context["state_pair_invariant_briefs"]),
        "status": (
            "PASS_CONTEXT_BOUND_TO_AUTHENTICATED_CALLS"
            if context_bound
            else "READY_NEXT_LIVE_RUN_NOT_YET_EVIDENCED"
        ),
        "typed_formula_kinds": len(creative_context["typed_formula_kinds"]),
        "uncertain_invariant_briefs": len(creative_context["uncertain_invariant_briefs"]),
    }


def _load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"app_id", "claude", "components", "release_policy", "schema_version"}:
        raise CoreCreativeDiscoveryError("core creative discovery config keys changed")
    if value["schema_version"] != CONFIG_SCHEMA or value["app_id"] != (
        "invariant.core-creative-discovery"
    ):
        raise CoreCreativeDiscoveryError("core creative discovery config identity changed")
    claude = value["claude"]
    if set(claude) != {
        "credential_env_var",
        "available_creative_roles",
        "required_completed_calls",
        "required_executable_hypotheses",
        "required_first_principles_prompt_context",
        "required_model",
        "required_roles",
    }:
        raise CoreCreativeDiscoveryError("core Claude policy keys changed")
    if (
        claude["credential_env_var"] != "ANTHROPIC_API_KEY"
        or claude["required_completed_calls"] < 8
        or claude["required_executable_hypotheses"] < 1
        or claude["required_first_principles_prompt_context"] is not True
        or set(claude["required_roles"]) != {"proposer", "critic"}
        or set(claude["available_creative_roles"]) != {role.value for role in ClaudeRole}
    ):
        raise CoreCreativeDiscoveryError("core Claude participation policy is too weak")
    policy = value["release_policy"]
    if (
        policy.get("required_serious_claim_backends")
        != ["exact_arithmetic", "cas", "smt", "interval", "lean"]
        or policy.get("backend_wrong_formula_mutation_required") is not True
        or policy.get("human_prior_art_review_required") is not True
        or policy.get("minimum_independent_level5_passes_before_open_problem", 0) < 3
    ):
        raise CoreCreativeDiscoveryError("core release policy is too weak")
    if set(value["components"]) != {
        "component_knockout_preflight_receipt",
        "claim_specific_prior_art_portfolio_preflight_receipt",
        "claim_specific_prior_art_portfolio_receipt",
        "dataset_challenge_receipt",
        "declarative_operational_receipt",
        "external_dataset_challenge_receipt",
        "external_structured_benchmark_receipt",
        "expanded_typed_grammar_receipt",
        "live_evidence_output",
        "learned_invariant_discovery_receipt",
        "level5_success_admission_receipt",
        "multi_host_reproduction_receipt",
        "proof_plan_search_receipt",
        "retained_piecewise_descendant_campaign_receipt",
        "retained_piecewise_replay_receipt",
        "serious_claim_verification_ladder_receipt",
        "state_pair_invariant_discovery_receipt",
        "symmetry_dimension_derivation_receipt",
        "uncertain_invariant_discovery_receipt",
    }:
        raise CoreCreativeDiscoveryError("core component bindings changed")
    return value


def _load_bound_receipts(
    root: Path, config: Mapping[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    components = config["components"]
    dataset_path = root / components["dataset_challenge_receipt"]
    external_dataset_path = root / components["external_dataset_challenge_receipt"]
    external_structured_path = components["external_structured_benchmark_receipt"]
    operational_path = root / components["declarative_operational_receipt"]
    multi_host_path = root / components["multi_host_reproduction_receipt"]
    expanded_grammar_path = root / components["expanded_typed_grammar_receipt"]
    proof_plan_path = root / components["proof_plan_search_receipt"]
    piecewise_replay_path = root / components["retained_piecewise_replay_receipt"]
    serious_claim_ladder_path = root / components["serious_claim_verification_ladder_receipt"]
    symmetry_dimension_path = root / components["symmetry_dimension_derivation_receipt"]
    component_knockout_path = root / components["component_knockout_preflight_receipt"]
    learned_invariant_path = root / components["learned_invariant_discovery_receipt"]
    level5_admission_path = root / components["level5_success_admission_receipt"]
    state_pair_invariant_path = root / components["state_pair_invariant_discovery_receipt"]
    uncertain_invariant_path = root / components["uncertain_invariant_discovery_receipt"]
    piecewise_descendant_path = root / components["retained_piecewise_descendant_campaign_receipt"]
    prior_art_portfolio_preflight_path = (
        root / components["claim_specific_prior_art_portfolio_preflight_receipt"]
    )
    prior_art_portfolio_path = root / components["claim_specific_prior_art_portfolio_receipt"]
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    external_dataset = json.loads(external_dataset_path.read_text(encoding="utf-8"))
    _, _, external_structured = load_stored_pack(root, external_structured_path)
    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    multi_host = json.loads(multi_host_path.read_text(encoding="utf-8"))
    expanded_grammar = json.loads(expanded_grammar_path.read_text(encoding="utf-8"))
    proof_plan_search = json.loads(proof_plan_path.read_text(encoding="utf-8"))
    piecewise_replay = json.loads(piecewise_replay_path.read_text(encoding="utf-8"))
    serious_claim_ladder = json.loads(serious_claim_ladder_path.read_text(encoding="utf-8"))
    symmetry_dimension = json.loads(symmetry_dimension_path.read_text(encoding="utf-8"))
    component_knockout = json.loads(component_knockout_path.read_text(encoding="utf-8"))
    learned_invariants = json.loads(learned_invariant_path.read_text(encoding="utf-8"))
    level5_admission = json.loads(level5_admission_path.read_text(encoding="utf-8"))
    state_pair_invariants = json.loads(state_pair_invariant_path.read_text(encoding="utf-8"))
    uncertain_invariants = json.loads(uncertain_invariant_path.read_text(encoding="utf-8"))
    piecewise_descendants = json.loads(piecewise_descendant_path.read_text(encoding="utf-8"))
    prior_art_portfolio_preflight = json.loads(
        prior_art_portfolio_preflight_path.read_text(encoding="utf-8")
    )
    prior_art_portfolio = json.loads(prior_art_portfolio_path.read_text(encoding="utf-8"))
    validate_dataset_challenges(dataset, root)
    validate_external_dataset_challenges(external_dataset, root)
    validate_operational_receipt(operational, root)
    validate_multi_host_receipt(multi_host, root)
    validate_expanded_grammar_receipt(expanded_grammar, root)
    validate_proof_plan_search(proof_plan_search, root)
    validate_piecewise_replay(piecewise_replay, root)
    validate_serious_claim_ladder(serious_claim_ladder, root)
    validate_symmetry_dimension_derivation(symmetry_dimension, root)
    validate_component_knockout_preflight(component_knockout, root)
    validate_learned_invariants(learned_invariants, root)
    validate_level5_success_admission(level5_admission, root)
    validate_state_pair_invariants(state_pair_invariants, root)
    validate_uncertain_invariants(uncertain_invariants, root)
    validate_piecewise_descendant_campaign(piecewise_descendants, root)
    validate_prior_art_portfolio_preflight(prior_art_portfolio_preflight, root)
    validate_prior_art_portfolio_receipt(prior_art_portfolio, prior_art_portfolio_preflight, root)
    return (
        operational,
        multi_host,
        expanded_grammar,
        dataset,
        external_dataset,
        external_structured,
        symmetry_dimension,
        proof_plan_search,
        piecewise_replay,
        serious_claim_ladder,
        component_knockout,
        learned_invariants,
        state_pair_invariants,
        uncertain_invariants,
        piecewise_descendants,
        prior_art_portfolio_preflight,
        prior_art_portfolio,
        level5_admission,
    )


def _claude_execution_summary(campaign: Mapping[str, Any]) -> dict[str, Any]:
    records: list[Mapping[str, Any]] = []
    contributions: list[Mapping[str, Any]] = []
    for benchmark in campaign.get("benchmarks", []):
        records.extend(benchmark.get("proposer_admission", {}).get("records", []))
        contributions.append(benchmark.get("claude_contribution", {}))
    origin_counts: dict[str, int] = {}
    for record in records:
        origin = str(record.get("llm_self_assessed_origin", "missing"))
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
    measured = [
        item
        for item in contributions
        if item.get("status") == "MEASURED_EXECUTABLE_CLAUDE_CONTRIBUTION"
    ]
    profiles_match = bool(measured) and all(
        item.get("grammar_depth_match") is True
        and item.get("evaluation_runtime_budget_match") is True
        and item.get("verifier_budget_match") is True
        for item in measured
    )
    return {
        "admission_records_sha256": canonical_sha256(records),
        "admitted_executable_hypotheses": sum(
            record.get("status") == "ADMITTED_EXECUTABLE" for record in records
        ),
        "behavior_novelty_against_deterministic_count": sum(
            int(item.get("behavior_novelty_against_deterministic_count", 0))
            for item in contributions
        ),
        "llm_self_assessed_origin_counts": dict(sorted(origin_counts.items())),
        "matched_control_profiles_verified": profiles_match,
        "non_executable_hypotheses_retained": sum(
            record.get("status") == "RETAINED_NON_EXECUTABLE" for record in records
        ),
        "retained_unscored_executable_candidates": sum(
            int(item.get("retained_unscored_executable_candidates", 0)) for item in contributions
        ),
        "proof_mechanism_novelty_against_deterministic_count": sum(
            int(item.get("proof_mechanism_novelty_against_deterministic_count", 0))
            for item in contributions
        ),
        "scored_executable_candidates": sum(
            int(item.get("scored_executable_candidates", 0)) for item in contributions
        ),
    }


def _validate_live_campaign(
    campaign: Mapping[str, Any],
    config: Mapping[str, Any],
    creative_context: Mapping[str, Any],
    root: Path,
) -> None:
    validate_creative_prompt_context(creative_context)
    claude = campaign.get("claude", {})
    calls = claude.get("calls", [])
    policy = config["claude"]
    roles = {call.get("role") for call in calls if call.get("status") == "completed"}
    models = {
        call.get("evidence", {}).get("model") for call in calls if call.get("status") == "completed"
    }
    structured = {
        call.get("evidence", {}).get("model_evidence", {}).get("structured_outputs_supported")
        for call in calls
        if call.get("status") == "completed"
    }
    context_bindings = {
        call.get("evidence", {}).get("creative_context_sha256")
        for call in calls
        if call.get("status") == "completed"
    }
    context_injected = {
        call.get("evidence", {}).get("creative_context_injected")
        for call in calls
        if call.get("status") == "completed"
    }
    execution = _claude_execution_summary(campaign)
    source_bindings = campaign.get("config", {})
    failures = []
    checks = {
        "campaign_schema": campaign.get("schema_version")
        == "invariant-external-creativity-validation-result-1.2",
        "campaign_substantive": campaign.get("claims", {}).get("claude_used_throughout") is True,
        "claude_status": claude.get("status") == "PASS",
        "completed_call_count": claude.get("completed_calls") == policy["required_completed_calls"],
        "proposer_hypotheses_present": claude.get("proposer_hypotheses", 0) >= 1,
        "call_trace_count": len(calls) == policy["required_completed_calls"],
        "required_roles": roles == set(policy["required_roles"]),
        "required_model": models == {policy["required_model"]},
        "structured_outputs": structured == {True},
        "prompt_context_binding": context_bindings == {creative_context["content_sha256"]},
        "prompt_context_injected": context_injected == {True},
        "executable_hypothesis_admitted": execution["admitted_executable_hypotheses"]
        >= policy["required_executable_hypotheses"],
        "executable_candidate_scored": execution["scored_executable_candidates"] >= 1,
        "matched_control_profiles": execution["matched_control_profiles_verified"] is True,
        "campaign_source_binding": source_bindings.get("source_sha256")
        == _normalized_file_sha256(root / EXTERNAL_CAMPAIGN_SOURCE_PATH),
        "claude_source_binding": source_bindings.get("claude_source_sha256")
        == _normalized_file_sha256(root / CLAUDE_API_SOURCE_PATH),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    if failures:
        counts = {
            "admitted": execution["admitted_executable_hypotheses"],
            "calls": claude.get("completed_calls", 0),
            "hypotheses": claude.get("proposer_hypotheses", 0),
            "scored": execution["scored_executable_candidates"],
        }
        raise CoreCreativeDiscoveryError(
            "authenticated Claude core participation failed: "
            f"{','.join(failures)}; counts={json.dumps(counts, sort_keys=True)}"
        )


def run_core(
    root: Path,
    *,
    credential_file: Path | None = None,
    campaign_runner: CampaignRunner | None = None,
) -> dict[str, Any]:
    """Execute a live core run and return only sanitized, sealed evidence."""

    root = root.resolve()
    config = _load_config(root)
    (
        operational,
        multi_host,
        expanded_grammar,
        dataset_challenges,
        external_dataset_challenges,
        external_structured_benchmarks,
        symmetry_dimension_derivation,
        proof_plan_search,
        piecewise_replay,
        serious_claim_ladder,
        component_knockout,
        learned_invariants,
        state_pair_invariants,
        uncertain_invariants,
        piecewise_descendants,
        prior_art_portfolio_preflight,
        prior_art_portfolio,
        level5_admission,
    ) = _load_bound_receipts(root, config)
    creative_prompt_context = build_creative_prompt_context(
        symmetry_dimension_derivation,
        learned_invariants,
        state_pair_invariants,
        uncertain_invariants,
        expanded_grammar,
        proof_plan_search,
    )
    environment = None
    if credential_file is not None:
        environment = dict(os.environ)
        environment["INVARIANT_ENV_FILE"] = str(credential_file.resolve())

    def live_runner(project_root: Path, context: Mapping[str, Any]) -> Mapping[str, Any]:
        return run_campaign(
            project_root,
            live_claude=True,
            claude_transport=FirstPrinciplesContextTransport(context),
            attempt_journal_path=project_root / LIVE_CALL_JOURNAL_PATH,
        )

    runner = campaign_runner or live_runner
    try:
        with activated_credential(
            project_root=root,
            env_var=config["claude"]["credential_env_var"],
            environment=environment,
        ) as activation:
            campaign = dict(runner(root, creative_prompt_context))
            try:
                _validate_live_campaign(campaign, config, creative_prompt_context, root)
            except CoreCreativeDiscoveryError:
                failed_output = root / FAILED_CAMPAIGN_PATH
                failed_output.parent.mkdir(parents=True, exist_ok=True)
                failed_output.write_text(
                    json.dumps(campaign, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise
            claude_execution = _claude_execution_summary(campaign)
            idea_archive = build_idea_archive(campaign)
            creative_expansion = build_creative_expansion(idea_archive, proof_plan_search)
            live_evidence = build_evidence_from_receipt(
                campaign, source_file_sha256=_serialized_sha256(campaign)
            )
    except CredentialActivationError as error:
        raise CoreCreativeDiscoveryError(str(error)) from error
    validate_evidence(live_evidence)
    level5 = level5_admission["summary"][
        "admitted_independently_reproduced_level5_successes"
    ]
    prior_art_reviews = [
        benchmark.get("prior_art", {}).get("human_review", {}).get("status")
        for benchmark in campaign.get("benchmarks", [])
    ]
    serious_claims = sum(
        bool(benchmark.get("claims", {}).get("serious_claim_released"))
        for benchmark in campaign.get("benchmarks", [])
    )
    prior_art_portfolio_claims = prior_art_portfolio["batch"]["completed_claims"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "app_id": config["app_id"],
        "source_bindings": {
            "core_application_source": {
                "path": SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / SOURCE_PATH),
            },
            "claude_api_source": {
                "path": CLAUDE_API_SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / CLAUDE_API_SOURCE_PATH),
            },
            "external_campaign_source": {
                "path": EXTERNAL_CAMPAIGN_SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / EXTERNAL_CAMPAIGN_SOURCE_PATH),
            },
            "config": {
                "path": CONFIG_PATH,
                "sha256": _normalized_file_sha256(root / CONFIG_PATH),
            },
            "prompt_context_source": {
                "path": PROMPT_CONTEXT_SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / PROMPT_CONTEXT_SOURCE_PATH),
            },
            "component_knockout_preflight_receipt": {
                "content_sha256": component_knockout["content_sha256"],
                "path": config["components"]["component_knockout_preflight_receipt"],
            },
            "claim_specific_prior_art_portfolio_preflight_receipt": {
                "content_sha256": prior_art_portfolio_preflight["content_sha256"],
                "path": config["components"][
                    "claim_specific_prior_art_portfolio_preflight_receipt"
                ],
            },
            "claim_specific_prior_art_portfolio_receipt": {
                "content_sha256": prior_art_portfolio["content_sha256"],
                "path": config["components"]["claim_specific_prior_art_portfolio_receipt"],
            },
            "dataset_challenge_receipt": {
                "content_sha256": dataset_challenges["content_sha256"],
                "path": config["components"]["dataset_challenge_receipt"],
            },
            "declarative_operational_receipt": {
                "content_sha256": operational["content_sha256"],
                "path": config["components"]["declarative_operational_receipt"],
            },
            "external_dataset_challenge_receipt": {
                "content_sha256": external_dataset_challenges["content_sha256"],
                "path": config["components"]["external_dataset_challenge_receipt"],
            },
            "external_structured_benchmark_receipt": {
                "content_sha256": external_structured_benchmarks["content_sha256"],
                "path": config["components"]["external_structured_benchmark_receipt"],
            },
            "learned_invariant_discovery_receipt": {
                "content_sha256": learned_invariants["content_sha256"],
                "path": config["components"]["learned_invariant_discovery_receipt"],
            },
            "level5_success_admission_receipt": {
                "content_sha256": level5_admission["content_sha256"],
                "path": config["components"]["level5_success_admission_receipt"],
            },
            "state_pair_invariant_discovery_receipt": {
                "content_sha256": state_pair_invariants["content_sha256"],
                "path": config["components"]["state_pair_invariant_discovery_receipt"],
            },
            "uncertain_invariant_discovery_receipt": {
                "content_sha256": uncertain_invariants["content_sha256"],
                "path": config["components"]["uncertain_invariant_discovery_receipt"],
            },
            "symmetry_dimension_derivation_receipt": {
                "content_sha256": symmetry_dimension_derivation["content_sha256"],
                "path": config["components"]["symmetry_dimension_derivation_receipt"],
            },
            "multi_host_reproduction_receipt": {
                "content_sha256": multi_host["content_sha256"],
                "path": config["components"]["multi_host_reproduction_receipt"],
            },
            "expanded_typed_grammar_receipt": {
                "content_sha256": expanded_grammar["content_sha256"],
                "path": config["components"]["expanded_typed_grammar_receipt"],
            },
            "proof_plan_search_receipt": {
                "content_sha256": proof_plan_search["content_sha256"],
                "path": config["components"]["proof_plan_search_receipt"],
            },
            "retained_piecewise_descendant_campaign_receipt": {
                "content_sha256": piecewise_descendants["content_sha256"],
                "path": config["components"]["retained_piecewise_descendant_campaign_receipt"],
            },
            "retained_piecewise_replay_receipt": {
                "content_sha256": piecewise_replay["content_sha256"],
                "path": config["components"]["retained_piecewise_replay_receipt"],
            },
            "serious_claim_verification_ladder_receipt": {
                "content_sha256": serious_claim_ladder["content_sha256"],
                "path": config["components"]["serious_claim_verification_ladder_receipt"],
            },
        },
        "credential_activation": activation.to_evidence(),
        "claude_runtime": {
            "authenticated_messages_api_working": True,
            "available_creative_roles": sorted(config["claude"]["available_creative_roles"]),
            "completed_calls": live_evidence["usage"]["calls"],
            "executable_contribution": claude_execution,
            "evidence": live_evidence,
            "model": config["claude"]["required_model"],
            "roles_completed": sorted(config["claude"]["required_roles"]),
            "status": "PASS_REQUIRED_CORE_PARTICIPATION",
        },
        "llm_prompt_context": _prompt_context_runtime_evidence(
            creative_prompt_context, live_evidence
        ),
        "idea_lineage_archive": idea_archive,
        "creative_expansion": creative_expansion,
        "component_knockout_preflight": component_knockout,
        "claim_specific_prior_art_portfolio_preflight": prior_art_portfolio_preflight,
        "claim_specific_prior_art_portfolio": prior_art_portfolio,
        "dataset_challenges": dataset_challenges,
        "external_dataset_challenges": external_dataset_challenges,
        "external_structured_benchmarks": external_structured_benchmarks,
        "learned_invariant_discovery": learned_invariants,
        "level5_success_admission": level5_admission,
        "state_pair_invariant_discovery": state_pair_invariants,
        "uncertain_invariant_discovery": uncertain_invariants,
        "symmetry_dimension_derivation": symmetry_dimension_derivation,
        "proof_plan_search": proof_plan_search,
        "retained_piecewise_descendant_campaign": piecewise_descendants,
        "retained_piecewise_descendant_claude_runtime": piecewise_descendants["claude_runtime"],
        "retained_piecewise_replay": piecewise_replay,
        "discovery_runtime": {
            "claim_specific_prior_art_automated_screens_complete": prior_art_portfolio[
                "release_gate"
            ]["all_automated_screens_complete"],
            "claim_specific_prior_art_claims": len(prior_art_portfolio_claims),
            "claim_specific_prior_art_external_request_budget": prior_art_portfolio["batch"][
                "cumulative_external_request_budget"
            ],
            "claim_specific_prior_art_named_human_reviews_complete": prior_art_portfolio[
                "release_gate"
            ]["all_named_human_reviews_complete"],
            "claim_specific_prior_art_no_exact_behavior_match_claims": sum(
                row["behavior_assessment"] == "NO_EXACT_MATCH_IN_QUERIED_RESULTS"
                for row in prior_art_portfolio_claims
            ),
            "claim_specific_prior_art_status": prior_art_portfolio["release_gate"]["status"],
            "declarative_extensions_admitted": len(
                operational["extension_admission"]["admitted_declarations"]
            ),
            "distinct_behavior_niches": operational["behavioral_archive"]["occupied_niches"],
            "proof_plan_closed": operational["proof_plan"]["closed"],
            "proof_plan_mechanisms": operational["proof_plan"]["mechanisms"],
            "typed_formula_kinds": expanded_grammar["summary"]["admitted_formula_kinds"],
            "typed_grammar_controls_passed": expanded_grammar["summary"]["controls_passed"],
            "typed_grammar_status": expanded_grammar["summary"]["status"],
            "dataset_challenge_kinds": dataset_challenges["summary"]["challenge_kinds"],
            "dataset_mutation_controls_rejected": dataset_challenges["summary"][
                "mutation_controls_rejected"
            ],
            "dataset_positive_controls_passed": dataset_challenges["summary"][
                "positive_controls_passed"
            ],
            "dataset_challenge_status": dataset_challenges["summary"]["status"],
            "external_dataset_challenge_kinds": external_dataset_challenges["summary"][
                "challenge_kinds"
            ],
            "external_dataset_challenges_passed": external_dataset_challenges["summary"][
                "external_challenges_passed"
            ],
            "external_dataset_mutation_controls_rejected": external_dataset_challenges["summary"][
                "mutation_controls_rejected"
            ],
            "external_dataset_status": external_dataset_challenges["summary"]["status"],
            "external_structured_benchmark_families": sorted(
                external_structured_benchmarks["coverage"]["representation_counts"]
            ),
            "external_structured_benchmark_level5_eligible": (
                external_structured_benchmarks["release_gate"]["level5_eligible"]
            ),
            "external_structured_benchmark_status": external_structured_benchmarks["release_gate"][
                "status"
            ],
            "external_structured_benchmark_tasks": external_structured_benchmarks["coverage"][
                "tasks"
            ],
            "first_principles_d4_basis_collapse_mutations_rejected": (
                symmetry_dimension_derivation["summary"]["basis_collapse_mutations_rejected"]
            ),
            "first_principles_d4_controls_passed": symmetry_dimension_derivation["summary"][
                "controls_passed"
            ],
            "first_principles_d4_dimension_mutations_rejected": (
                symmetry_dimension_derivation["summary"]["dimension_mutations_rejected"]
            ),
            "first_principles_d4_invariant_coordinates": symmetry_dimension_derivation["summary"][
                "invariant_coordinates"
            ],
            "first_principles_d4_multi_coordinate_controls": symmetry_dimension_derivation[
                "summary"
            ]["multi_coordinate_controls"],
            "first_principles_d4_status": symmetry_dimension_derivation["summary"]["status"],
            "first_principles_d4_symmetry_mutations_rejected": (
                symmetry_dimension_derivation["summary"]["symmetry_mutations_rejected"]
            ),
            "learned_invariant_deployment_repaired_coordinates": learned_invariants["summary"][
                "deployment_repaired_coordinates"
            ],
            "learned_invariant_identified_passes": learned_invariants["summary"][
                "identified_passes"
            ],
            "learned_invariant_problems": learned_invariants["summary"]["problems"],
            "learned_invariant_shift_rejections": learned_invariants["summary"]["shift_rejections"],
            "learned_invariant_status": learned_invariants["summary"]["status"],
            "learned_invariant_training_coordinates_retained": learned_invariants["summary"][
                "training_coordinates_retained"
            ],
            "learned_invariant_underdetermined_controls": learned_invariants["summary"][
                "underdetermined_controls"
            ],
            "level5_admission_status": level5_admission["summary"]["status"],
            "level5_campaign_local_process_passes": level5_admission["summary"][
                "campaign_local_level5_process_passes"
            ],
            "level5_process_passes_before_external_signature": level5_admission["summary"][
                "process_passes_before_external_signature"
            ],
            "state_pair_algebraically_independent_coordinates": state_pair_invariants["summary"][
                "algebraically_independent_coordinates"
            ],
            "state_pair_controls": state_pair_invariants["summary"]["controls"],
            "state_pair_deployment_failures": state_pair_invariants["summary"][
                "deployment_failures"
            ],
            "state_pair_feature_grammar_kinds": state_pair_invariants["summary"][
                "feature_grammar_kinds"
            ],
            "state_pair_higher_degree_controls": state_pair_invariants["summary"][
                "higher_degree_controls"
            ],
            "state_pair_matrix_action_controls": state_pair_invariants["summary"][
                "matrix_action_controls"
            ],
            "state_pair_multivariate_rational_action_controls": state_pair_invariants["summary"][
                "multivariate_rational_action_controls"
            ],
            "state_pair_nonlinear_action_controls": state_pair_invariants["summary"][
                "nonlinear_action_controls"
            ],
            "state_pair_rational_action_controls": state_pair_invariants["summary"][
                "rational_action_controls"
            ],
            "state_pair_status": state_pair_invariants["summary"]["status"],
            "state_pair_target_blind_controls": state_pair_invariants["summary"][
                "target_blind_controls"
            ],
            "state_pair_transcendental_action_controls": state_pair_invariants["summary"][
                "transcendental_action_controls"
            ],
            "uncertain_invariant_censored_controls": uncertain_invariants["summary"][
                "censored_controls"
            ],
            "uncertain_invariant_controls": uncertain_invariants["summary"]["controls"],
            "uncertain_invariant_dependent_joint_controls": uncertain_invariants["summary"][
                "dependent_joint_controls"
            ],
            "uncertain_invariant_deployment_failed_candidates": uncertain_invariants["summary"][
                "deployment_failed_candidates"
            ],
            "uncertain_invariant_deployment_surviving_candidates": uncertain_invariants["summary"][
                "deployment_surviving_candidates"
            ],
            "uncertain_invariant_missingness_controls": uncertain_invariants["summary"][
                "missingness_controls"
            ],
            "uncertain_invariant_marginal_false_positives_rejected": uncertain_invariants[
                "summary"
            ]["marginal_false_positive_candidates_rejected"],
            "uncertain_invariant_noisy_controls": uncertain_invariants["summary"]["noisy_controls"],
            "uncertain_invariant_status": uncertain_invariants["summary"]["status"],
            "uncertain_invariant_training_candidates_retained": uncertain_invariants["summary"][
                "training_candidates_retained"
            ],
            "uncertain_invariant_unit_hypothesis_branches_retained": uncertain_invariants[
                "summary"
            ]["unit_hypothesis_branches_retained"],
            "uncertain_invariant_unit_uncertainty_controls": uncertain_invariants["summary"][
                "unit_uncertainty_controls"
            ],
            "independent_proof_plan_mechanisms": proof_plan_search["summary"]["mechanisms"],
            "independent_proof_plan_mutations_rejected": proof_plan_search["summary"][
                "mutation_controls_rejected"
            ],
            "independent_proof_plan_routes_closed": proof_plan_search["summary"][
                "positive_routes_closed"
            ],
            "independent_proof_plan_status": proof_plan_search["summary"]["status"],
            "retained_piecewise_admitted": piecewise_replay["summary"][
                "admitted_by_current_executor"
            ],
            "retained_piecewise_exact_agreements": piecewise_replay["summary"][
                "exact_primary_independent_agreements"
            ],
            "retained_piecewise_origin_counts": piecewise_replay["summary"][
                "llm_self_assessed_origin_counts"
            ],
            "retained_piecewise_resource_matched_controls": piecewise_replay["summary"][
                "resource_matched_controls"
            ],
            "retained_piecewise_replay_status": piecewise_replay["summary"]["status"],
            "retained_piecewise_train_exact_holdout_failed": piecewise_replay["summary"][
                "train_exact_holdout_failed"
            ],
            "retained_piecewise_zero_holdout_bounded_unknown": piecewise_replay["summary"][
                "zero_holdout_loss_bounded_unknown"
            ],
            "retained_piecewise_descendant_admitted": piecewise_descendants["summary"][
                "admitted_executable_descendants"
            ],
            "retained_piecewise_descendant_calls": piecewise_descendants["claude_runtime"][
                "completed_calls"
            ],
            "retained_piecewise_descendant_exact_agreements": piecewise_descendants["summary"][
                "exact_primary_independent_agreements"
            ],
            "retained_piecewise_descendant_ideas": piecewise_descendants["summary"][
                "descendant_ideas_retained"
            ],
            "retained_piecewise_descendant_nonexecutable_retained": piecewise_descendants[
                "summary"
            ]["nonexecutable_descendants_retained"],
            "retained_piecewise_descendant_parent_branches_preserved": piecewise_descendants[
                "summary"
            ]["parent_branches_preserved"],
            "retained_piecewise_descendant_resource_matched_controls": piecewise_descendants[
                "summary"
            ]["resource_matched_controls"],
            "retained_piecewise_descendant_status": piecewise_descendants["summary"]["status"],
            "retained_piecewise_descendant_zero_holdout_bounded_unknown": (
                piecewise_descendants["summary"]["zero_fresh_holdout_bounded_unknown"]
            ),
            "retained_piecewise_descendant_zero_holdout_known_control": (
                piecewise_descendants["summary"]["zero_fresh_holdout_known_control"]
            ),
            "component_knockout_experiments_preflighted": component_knockout["design"][
                "experiments"
            ],
            "component_knockout_live_runs_complete": component_knockout["release_gate"][
                "component_knockout_live_runs_complete"
            ],
            "component_knockout_preflight_status": component_knockout["release_gate"]["status"],
            "component_knockout_scheduled_slots": component_knockout["schedule"][
                "total_scheduled_slots"
            ],
        },
        "verification": {
            "backends_required_for_serious_claim": config["release_policy"][
                "required_serious_claim_backends"
            ],
            "multi_host_status": multi_host["reproduction"]["status"],
            "received_machines": multi_host["reproduction"]["received_machines"],
            "lean_kernel_checked": multi_host["lean"]["kernel_checked"],
            "serious_claim_ladder_status": serious_claim_ladder["summary"]["status"],
            "serious_claim_backend_mutations_rejected": serious_claim_ladder["summary"][
                "backend_mathematical_mutations_rejected"
            ],
            "serious_claim_lean_mutation_artifact_bound": serious_claim_ladder["summary"][
                "lean_kernel_mutation_artifact_bound"
            ],
            "serious_claim_required_stage_order": serious_claim_ladder["summary"][
                "required_stage_order"
            ],
            "serious_claims_released_by_ladder": serious_claim_ladder["release_gate"][
                "serious_claims_released"
            ],
        },
        "release_gate": {
            "claim_specific_prior_art_automated_screens_complete": prior_art_portfolio[
                "release_gate"
            ]["all_automated_screens_complete"],
            "claim_specific_prior_art_named_human_reviews_complete": prior_art_portfolio[
                "release_gate"
            ]["all_named_human_reviews_complete"],
            "component_knockout_live_runs_complete": component_knockout["release_gate"][
                "component_knockout_live_runs_complete"
            ],
            "human_prior_art_reviews_complete": all(
                status == "COMPLETED" for status in prior_art_reviews
            )
            and prior_art_portfolio["release_gate"]["all_named_human_reviews_complete"],
            "llm_first_principles_lane_live_run_complete": True,
            "retained_piecewise_descendant_live_run_complete": True,
            "level5_process_passes": level5,
            "minimum_level5_process_passes": config["release_policy"][
                "minimum_independent_level5_passes_before_open_problem"
            ],
            "open_problem_authorized": level5_admission["release_gate"][
                "open_problem_authorized"
            ],
            "serious_claims_released": serious_claims,
            "status": (
                "READY_FAMOUS_OPEN_PROBLEM_PREREGISTRATION"
                if level5_admission["release_gate"]["open_problem_authorized"]
                else "BLOCKED_CALIBRATION_OR_HUMAN_REVIEW_INCOMPLETE"
            ),
        },
        "claims": {
            "claude_is_verifier_authority": False,
            "credential_material_persisted": False,
            "llm_origin_assessment_is_novelty_authority": False,
            "novel_formula_established": False,
            "open_problem_solved": False,
        },
    }
    if (
        serious_claims != 0
        or body["claims"]["credential_material_persisted"] is not False
    ):
        raise CoreCreativeDiscoveryError("core release boundary opened unexpectedly")
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root)
    return body


def rebind_core_receipt(root: Path, previous: Mapping[str, Any]) -> dict[str, Any]:
    """Rebind sanitized live LLM evidence after deterministic gate receipts advance."""

    root = root.resolve()
    previous_body = {key: item for key, item in previous.items() if key != "content_sha256"}
    if (
        previous.get("content_sha256") != canonical_sha256(previous_body)
        or previous.get("schema_version")
        not in {
            "invariant-core-creative-discovery-runtime-1.3",
            "invariant-core-creative-discovery-runtime-1.4",
            "invariant-core-creative-discovery-runtime-1.5",
            "invariant-core-creative-discovery-runtime-1.6",
            "invariant-core-creative-discovery-runtime-1.7",
            "invariant-core-creative-discovery-runtime-1.8",
            "invariant-core-creative-discovery-runtime-1.9",
            "invariant-core-creative-discovery-runtime-2.0",
            "invariant-core-creative-discovery-runtime-2.1",
            "invariant-core-creative-discovery-runtime-2.2",
            "invariant-core-creative-discovery-runtime-2.3",
            "invariant-core-creative-discovery-runtime-2.4",
            "invariant-core-creative-discovery-runtime-2.5",
            "invariant-core-creative-discovery-runtime-2.6",
            "invariant-core-creative-discovery-runtime-2.7",
            "invariant-core-creative-discovery-runtime-2.8",
            "invariant-core-creative-discovery-runtime-2.9",
            "invariant-core-creative-discovery-runtime-3.0",
            SCHEMA_VERSION,
        }
        or previous.get("app_id") != "invariant.core-creative-discovery"
    ):
        raise CoreCreativeDiscoveryError(
            "previous core runtime receipt is not a sealed predecessor"
        )
    runtime = previous.get("claude_runtime", {})
    credential = previous.get("credential_activation", {})
    if (
        runtime.get("status") != "PASS_REQUIRED_CORE_PARTICIPATION"
        or runtime.get("authenticated_messages_api_working") is not True
        or runtime.get("completed_calls", 0) < 8
        or runtime.get("executable_contribution", {}).get("scored_executable_candidates", 0) < 1
        or runtime.get("executable_contribution", {}).get("matched_control_profiles_verified")
        is not True
        or credential.get("credential_persisted") is not False
        or credential.get("credential_value_recorded") is not False
    ):
        raise CoreCreativeDiscoveryError("previous core LLM evidence is not reusable")
    validate_evidence(runtime["evidence"])
    config = _load_config(root)
    (
        operational,
        multi_host,
        expanded_grammar,
        dataset_challenges,
        external_dataset_challenges,
        external_structured_benchmarks,
        symmetry_dimension_derivation,
        proof_plan_search,
        piecewise_replay,
        serious_claim_ladder,
        component_knockout,
        learned_invariants,
        state_pair_invariants,
        uncertain_invariants,
        piecewise_descendants,
        prior_art_portfolio_preflight,
        prior_art_portfolio,
        level5_admission,
    ) = _load_bound_receipts(root, config)
    creative_prompt_context = build_creative_prompt_context(
        symmetry_dimension_derivation,
        learned_invariants,
        state_pair_invariants,
        uncertain_invariants,
        expanded_grammar,
        proof_plan_search,
    )
    value = deepcopy(dict(previous))
    prior_art_portfolio_claims = prior_art_portfolio["batch"]["completed_claims"]
    value["schema_version"] = SCHEMA_VERSION
    value["source_bindings"] = {
        "core_application_source": {
            "path": SOURCE_PATH,
            "sha256": _normalized_file_sha256(root / SOURCE_PATH),
        },
        "claude_api_source": {
            "path": CLAUDE_API_SOURCE_PATH,
            "sha256": _normalized_file_sha256(root / CLAUDE_API_SOURCE_PATH),
        },
        "external_campaign_source": {
            "path": EXTERNAL_CAMPAIGN_SOURCE_PATH,
            "sha256": _normalized_file_sha256(root / EXTERNAL_CAMPAIGN_SOURCE_PATH),
        },
        "config": {"path": CONFIG_PATH, "sha256": _normalized_file_sha256(root / CONFIG_PATH)},
        "prompt_context_source": {
            "path": PROMPT_CONTEXT_SOURCE_PATH,
            "sha256": _normalized_file_sha256(root / PROMPT_CONTEXT_SOURCE_PATH),
        },
        "component_knockout_preflight_receipt": {
            "content_sha256": component_knockout["content_sha256"],
            "path": config["components"]["component_knockout_preflight_receipt"],
        },
        "claim_specific_prior_art_portfolio_preflight_receipt": {
            "content_sha256": prior_art_portfolio_preflight["content_sha256"],
            "path": config["components"]["claim_specific_prior_art_portfolio_preflight_receipt"],
        },
        "claim_specific_prior_art_portfolio_receipt": {
            "content_sha256": prior_art_portfolio["content_sha256"],
            "path": config["components"]["claim_specific_prior_art_portfolio_receipt"],
        },
        "dataset_challenge_receipt": {
            "content_sha256": dataset_challenges["content_sha256"],
            "path": config["components"]["dataset_challenge_receipt"],
        },
        "declarative_operational_receipt": {
            "content_sha256": operational["content_sha256"],
            "path": config["components"]["declarative_operational_receipt"],
        },
        "external_dataset_challenge_receipt": {
            "content_sha256": external_dataset_challenges["content_sha256"],
            "path": config["components"]["external_dataset_challenge_receipt"],
        },
        "external_structured_benchmark_receipt": {
            "content_sha256": external_structured_benchmarks["content_sha256"],
            "path": config["components"]["external_structured_benchmark_receipt"],
        },
        "learned_invariant_discovery_receipt": {
            "content_sha256": learned_invariants["content_sha256"],
            "path": config["components"]["learned_invariant_discovery_receipt"],
        },
        "level5_success_admission_receipt": {
            "content_sha256": level5_admission["content_sha256"],
            "path": config["components"]["level5_success_admission_receipt"],
        },
        "state_pair_invariant_discovery_receipt": {
            "content_sha256": state_pair_invariants["content_sha256"],
            "path": config["components"]["state_pair_invariant_discovery_receipt"],
        },
        "uncertain_invariant_discovery_receipt": {
            "content_sha256": uncertain_invariants["content_sha256"],
            "path": config["components"]["uncertain_invariant_discovery_receipt"],
        },
        "symmetry_dimension_derivation_receipt": {
            "content_sha256": symmetry_dimension_derivation["content_sha256"],
            "path": config["components"]["symmetry_dimension_derivation_receipt"],
        },
        "multi_host_reproduction_receipt": {
            "content_sha256": multi_host["content_sha256"],
            "path": config["components"]["multi_host_reproduction_receipt"],
        },
        "expanded_typed_grammar_receipt": {
            "content_sha256": expanded_grammar["content_sha256"],
            "path": config["components"]["expanded_typed_grammar_receipt"],
        },
        "proof_plan_search_receipt": {
            "content_sha256": proof_plan_search["content_sha256"],
            "path": config["components"]["proof_plan_search_receipt"],
        },
        "retained_piecewise_descendant_campaign_receipt": {
            "content_sha256": piecewise_descendants["content_sha256"],
            "path": config["components"]["retained_piecewise_descendant_campaign_receipt"],
        },
        "retained_piecewise_replay_receipt": {
            "content_sha256": piecewise_replay["content_sha256"],
            "path": config["components"]["retained_piecewise_replay_receipt"],
        },
        "serious_claim_verification_ladder_receipt": {
            "content_sha256": serious_claim_ladder["content_sha256"],
            "path": config["components"]["serious_claim_verification_ladder_receipt"],
        },
    }
    value["component_knockout_preflight"] = component_knockout
    value["claim_specific_prior_art_portfolio_preflight"] = prior_art_portfolio_preflight
    value["claim_specific_prior_art_portfolio"] = prior_art_portfolio
    value["dataset_challenges"] = dataset_challenges
    value["external_dataset_challenges"] = external_dataset_challenges
    value["external_structured_benchmarks"] = external_structured_benchmarks
    value["learned_invariant_discovery"] = learned_invariants
    value["level5_success_admission"] = level5_admission
    value["state_pair_invariant_discovery"] = state_pair_invariants
    value["uncertain_invariant_discovery"] = uncertain_invariants
    value["symmetry_dimension_derivation"] = symmetry_dimension_derivation
    value["proof_plan_search"] = proof_plan_search
    value["retained_piecewise_descendant_campaign"] = piecewise_descendants
    value["retained_piecewise_descendant_claude_runtime"] = piecewise_descendants["claude_runtime"]
    value["retained_piecewise_replay"] = piecewise_replay
    value["llm_prompt_context"] = _prompt_context_runtime_evidence(
        creative_prompt_context, runtime["evidence"]
    )
    value["discovery_runtime"] = {
        **value["discovery_runtime"],
        "claim_specific_prior_art_automated_screens_complete": prior_art_portfolio["release_gate"][
            "all_automated_screens_complete"
        ],
        "claim_specific_prior_art_claims": len(prior_art_portfolio_claims),
        "claim_specific_prior_art_external_request_budget": prior_art_portfolio["batch"][
            "cumulative_external_request_budget"
        ],
        "claim_specific_prior_art_named_human_reviews_complete": prior_art_portfolio[
            "release_gate"
        ]["all_named_human_reviews_complete"],
        "claim_specific_prior_art_no_exact_behavior_match_claims": sum(
            row["behavior_assessment"] == "NO_EXACT_MATCH_IN_QUERIED_RESULTS"
            for row in prior_art_portfolio_claims
        ),
        "claim_specific_prior_art_status": prior_art_portfolio["release_gate"]["status"],
        "component_knockout_experiments_preflighted": component_knockout["design"]["experiments"],
        "component_knockout_live_runs_complete": component_knockout["release_gate"][
            "component_knockout_live_runs_complete"
        ],
        "component_knockout_preflight_status": component_knockout["release_gate"]["status"],
        "component_knockout_scheduled_slots": component_knockout["schedule"][
            "total_scheduled_slots"
        ],
        "external_dataset_challenge_kinds": external_dataset_challenges["summary"][
            "challenge_kinds"
        ],
        "external_dataset_challenges_passed": external_dataset_challenges["summary"][
            "external_challenges_passed"
        ],
        "external_dataset_mutation_controls_rejected": external_dataset_challenges["summary"][
            "mutation_controls_rejected"
        ],
        "external_dataset_status": external_dataset_challenges["summary"]["status"],
        "external_structured_benchmark_families": sorted(
            external_structured_benchmarks["coverage"]["representation_counts"]
        ),
        "external_structured_benchmark_level5_eligible": external_structured_benchmarks[
            "release_gate"
        ]["level5_eligible"],
        "external_structured_benchmark_status": external_structured_benchmarks["release_gate"][
            "status"
        ],
        "external_structured_benchmark_tasks": external_structured_benchmarks["coverage"]["tasks"],
        "first_principles_d4_basis_collapse_mutations_rejected": symmetry_dimension_derivation[
            "summary"
        ]["basis_collapse_mutations_rejected"],
        "first_principles_d4_controls_passed": symmetry_dimension_derivation["summary"][
            "controls_passed"
        ],
        "first_principles_d4_dimension_mutations_rejected": symmetry_dimension_derivation[
            "summary"
        ]["dimension_mutations_rejected"],
        "first_principles_d4_invariant_coordinates": symmetry_dimension_derivation["summary"][
            "invariant_coordinates"
        ],
        "first_principles_d4_multi_coordinate_controls": symmetry_dimension_derivation["summary"][
            "multi_coordinate_controls"
        ],
        "first_principles_d4_status": symmetry_dimension_derivation["summary"]["status"],
        "first_principles_d4_symmetry_mutations_rejected": symmetry_dimension_derivation["summary"][
            "symmetry_mutations_rejected"
        ],
        "learned_invariant_deployment_repaired_coordinates": learned_invariants["summary"][
            "deployment_repaired_coordinates"
        ],
        "learned_invariant_identified_passes": learned_invariants["summary"]["identified_passes"],
        "learned_invariant_problems": learned_invariants["summary"]["problems"],
        "learned_invariant_shift_rejections": learned_invariants["summary"]["shift_rejections"],
        "learned_invariant_status": learned_invariants["summary"]["status"],
        "learned_invariant_training_coordinates_retained": learned_invariants["summary"][
            "training_coordinates_retained"
        ],
        "learned_invariant_underdetermined_controls": learned_invariants["summary"][
            "underdetermined_controls"
        ],
        "level5_admission_status": level5_admission["summary"]["status"],
        "level5_campaign_local_process_passes": level5_admission["summary"][
            "campaign_local_level5_process_passes"
        ],
        "level5_process_passes_before_external_signature": level5_admission["summary"][
            "process_passes_before_external_signature"
        ],
        "state_pair_algebraically_independent_coordinates": state_pair_invariants["summary"][
            "algebraically_independent_coordinates"
        ],
        "state_pair_controls": state_pair_invariants["summary"]["controls"],
        "state_pair_deployment_failures": state_pair_invariants["summary"]["deployment_failures"],
        "state_pair_feature_grammar_kinds": state_pair_invariants["summary"][
            "feature_grammar_kinds"
        ],
        "state_pair_higher_degree_controls": state_pair_invariants["summary"][
            "higher_degree_controls"
        ],
        "state_pair_matrix_action_controls": state_pair_invariants["summary"][
            "matrix_action_controls"
        ],
        "state_pair_multivariate_rational_action_controls": state_pair_invariants["summary"][
            "multivariate_rational_action_controls"
        ],
        "state_pair_nonlinear_action_controls": state_pair_invariants["summary"][
            "nonlinear_action_controls"
        ],
        "state_pair_rational_action_controls": state_pair_invariants["summary"][
            "rational_action_controls"
        ],
        "state_pair_status": state_pair_invariants["summary"]["status"],
        "state_pair_target_blind_controls": state_pair_invariants["summary"][
            "target_blind_controls"
        ],
        "state_pair_transcendental_action_controls": state_pair_invariants["summary"][
            "transcendental_action_controls"
        ],
        "uncertain_invariant_censored_controls": uncertain_invariants["summary"][
            "censored_controls"
        ],
        "uncertain_invariant_controls": uncertain_invariants["summary"]["controls"],
        "uncertain_invariant_dependent_joint_controls": uncertain_invariants["summary"][
            "dependent_joint_controls"
        ],
        "uncertain_invariant_deployment_failed_candidates": uncertain_invariants["summary"][
            "deployment_failed_candidates"
        ],
        "uncertain_invariant_deployment_surviving_candidates": uncertain_invariants["summary"][
            "deployment_surviving_candidates"
        ],
        "uncertain_invariant_missingness_controls": uncertain_invariants["summary"][
            "missingness_controls"
        ],
        "uncertain_invariant_marginal_false_positives_rejected": uncertain_invariants["summary"][
            "marginal_false_positive_candidates_rejected"
        ],
        "uncertain_invariant_noisy_controls": uncertain_invariants["summary"]["noisy_controls"],
        "uncertain_invariant_status": uncertain_invariants["summary"]["status"],
        "uncertain_invariant_training_candidates_retained": uncertain_invariants["summary"][
            "training_candidates_retained"
        ],
        "uncertain_invariant_unit_hypothesis_branches_retained": uncertain_invariants["summary"][
            "unit_hypothesis_branches_retained"
        ],
        "uncertain_invariant_unit_uncertainty_controls": uncertain_invariants["summary"][
            "unit_uncertainty_controls"
        ],
        "retained_piecewise_admitted": piecewise_replay["summary"]["admitted_by_current_executor"],
        "retained_piecewise_exact_agreements": piecewise_replay["summary"][
            "exact_primary_independent_agreements"
        ],
        "retained_piecewise_origin_counts": piecewise_replay["summary"][
            "llm_self_assessed_origin_counts"
        ],
        "retained_piecewise_resource_matched_controls": piecewise_replay["summary"][
            "resource_matched_controls"
        ],
        "retained_piecewise_replay_status": piecewise_replay["summary"]["status"],
        "retained_piecewise_train_exact_holdout_failed": piecewise_replay["summary"][
            "train_exact_holdout_failed"
        ],
        "retained_piecewise_zero_holdout_bounded_unknown": piecewise_replay["summary"][
            "zero_holdout_loss_bounded_unknown"
        ],
        "retained_piecewise_descendant_admitted": piecewise_descendants["summary"][
            "admitted_executable_descendants"
        ],
        "retained_piecewise_descendant_calls": piecewise_descendants["claude_runtime"][
            "completed_calls"
        ],
        "retained_piecewise_descendant_exact_agreements": piecewise_descendants["summary"][
            "exact_primary_independent_agreements"
        ],
        "retained_piecewise_descendant_ideas": piecewise_descendants["summary"][
            "descendant_ideas_retained"
        ],
        "retained_piecewise_descendant_nonexecutable_retained": piecewise_descendants["summary"][
            "nonexecutable_descendants_retained"
        ],
        "retained_piecewise_descendant_parent_branches_preserved": piecewise_descendants["summary"][
            "parent_branches_preserved"
        ],
        "retained_piecewise_descendant_resource_matched_controls": piecewise_descendants["summary"][
            "resource_matched_controls"
        ],
        "retained_piecewise_descendant_status": piecewise_descendants["summary"]["status"],
        "retained_piecewise_descendant_zero_holdout_bounded_unknown": piecewise_descendants[
            "summary"
        ]["zero_fresh_holdout_bounded_unknown"],
        "retained_piecewise_descendant_zero_holdout_known_control": piecewise_descendants[
            "summary"
        ]["zero_fresh_holdout_known_control"],
    }
    value["verification"] = {
        "backends_required_for_serious_claim": config["release_policy"][
            "required_serious_claim_backends"
        ],
        "multi_host_status": multi_host["reproduction"]["status"],
        "received_machines": multi_host["reproduction"]["received_machines"],
        "lean_kernel_checked": multi_host["lean"]["kernel_checked"],
        "serious_claim_ladder_status": serious_claim_ladder["summary"]["status"],
        "serious_claim_backend_mutations_rejected": serious_claim_ladder["summary"][
            "backend_mathematical_mutations_rejected"
        ],
        "serious_claim_lean_mutation_artifact_bound": serious_claim_ladder["summary"][
            "lean_kernel_mutation_artifact_bound"
        ],
        "serious_claim_required_stage_order": serious_claim_ladder["summary"][
            "required_stage_order"
        ],
        "serious_claims_released_by_ladder": serious_claim_ladder["release_gate"][
            "serious_claims_released"
        ],
    }
    value["release_gate"] = {
        **value["release_gate"],
        "claim_specific_prior_art_automated_screens_complete": prior_art_portfolio["release_gate"][
            "all_automated_screens_complete"
        ],
        "claim_specific_prior_art_named_human_reviews_complete": prior_art_portfolio[
            "release_gate"
        ]["all_named_human_reviews_complete"],
        "component_knockout_live_runs_complete": component_knockout["release_gate"][
            "component_knockout_live_runs_complete"
        ],
        "human_prior_art_reviews_complete": (
            value["release_gate"].get("human_prior_art_reviews_complete") is True
            and prior_art_portfolio["release_gate"]["all_named_human_reviews_complete"]
        ),
        "llm_first_principles_lane_live_run_complete": value["llm_prompt_context"][
            "authenticated_calls_bound_to_context"
        ],
        "retained_piecewise_descendant_live_run_complete": True,
        "level5_process_passes": level5_admission["summary"][
            "admitted_independently_reproduced_level5_successes"
        ],
        "minimum_level5_process_passes": level5_admission["summary"][
            "minimum_required_before_open_problem"
        ],
        "open_problem_authorized": level5_admission["release_gate"][
            "open_problem_authorized"
        ],
        "status": (
            "READY_FAMOUS_OPEN_PROBLEM_PREREGISTRATION"
            if level5_admission["release_gate"]["open_problem_authorized"]
            else "BLOCKED_CALIBRATION_OR_HUMAN_REVIEW_INCOMPLETE"
        ),
    }
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    validate_receipt(value, root)
    return value


def validate_receipt(value: Mapping[str, Any], root: Path | None = None) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CoreCreativeDiscoveryError("core runtime receipt content seal changed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CoreCreativeDiscoveryError("core runtime receipt schema changed")
    runtime = value.get("claude_runtime", {})
    credential = value.get("credential_activation", {})
    executable = runtime.get("executable_contribution", {})
    if (
        runtime.get("status") != "PASS_REQUIRED_CORE_PARTICIPATION"
        or runtime.get("authenticated_messages_api_working") is not True
        or runtime.get("completed_calls", 0) < 8
        or executable.get("admitted_executable_hypotheses", 0) < 1
        or executable.get("scored_executable_candidates", 0) < 1
        or executable.get("matched_control_profiles_verified") is not True
        or set(runtime.get("available_creative_roles", [])) != {role.value for role in ClaudeRole}
        or credential.get("credential_persisted") is not False
        or credential.get("credential_value_recorded") is not False
    ):
        raise CoreCreativeDiscoveryError("core Claude health evidence changed")
    validate_evidence(runtime["evidence"])
    live_calls = runtime["evidence"].get("calls", [])
    if any(
        call.get("wire_contract_adapter_used") is not (call.get("role") == "critic")
        for call in live_calls
    ):
        raise CoreCreativeDiscoveryError("core provider wire-contract evidence changed")
    validate_idea_archive(value.get("idea_lineage_archive", {}))
    proof_plan_search = value.get("proof_plan_search", {})
    validate_proof_plan_search(proof_plan_search, root)
    creative_prompt_context = build_creative_prompt_context(
        value.get("symmetry_dimension_derivation", {}),
        value.get("learned_invariant_discovery", {}),
        value.get("state_pair_invariant_discovery", {}),
        value.get("uncertain_invariant_discovery", {}),
        {
            "summary": {
                "admitted_formula_kinds": value.get("discovery_runtime", {}).get(
                    "typed_formula_kinds"
                )
            }
        },
        proof_plan_search,
    )
    expected_prompt_context_evidence = _prompt_context_runtime_evidence(
        creative_prompt_context, runtime["evidence"]
    )
    if value.get("llm_prompt_context") != expected_prompt_context_evidence:
        raise CoreCreativeDiscoveryError("core authenticated prompt context evidence changed")
    context_bound = expected_prompt_context_evidence["authenticated_calls_bound_to_context"]
    if (
        value.get("release_gate", {}).get("llm_first_principles_lane_live_run_complete")
        is not context_bound
    ):
        raise CoreCreativeDiscoveryError("core authenticated prompt context release gate changed")
    validate_creative_expansion(value.get("creative_expansion", {}), proof_plan_search)
    validate_piecewise_replay(value.get("retained_piecewise_replay", {}), root or Path.cwd())
    piecewise_descendants = value.get("retained_piecewise_descendant_campaign", {})
    validate_piecewise_descendant_campaign(piecewise_descendants, root or Path.cwd())
    if value.get("retained_piecewise_descendant_claude_runtime") != piecewise_descendants.get(
        "claude_runtime"
    ):
        raise CoreCreativeDiscoveryError("core descendant Claude runtime binding changed")
    validate_dataset_challenges(value.get("dataset_challenges", {}), root)
    validate_external_dataset_challenges(value.get("external_dataset_challenges", {}), root)
    validate_symmetry_dimension_derivation(
        value.get("symmetry_dimension_derivation", {}), root or Path.cwd()
    )
    validate_learned_invariants(value.get("learned_invariant_discovery", {}), root or Path.cwd())
    level5_admission = value.get("level5_success_admission", {})
    validate_level5_success_admission(level5_admission, root or Path.cwd())
    validate_state_pair_invariants(
        value.get("state_pair_invariant_discovery", {}), root or Path.cwd()
    )
    validate_uncertain_invariants(
        value.get("uncertain_invariant_discovery", {}), root or Path.cwd()
    )
    external_structured = value.get("external_structured_benchmarks", {})
    external_structured_body = {
        key: item for key, item in external_structured.items() if key != "content_sha256"
    }
    if external_structured.get(
        "schema_version"
    ) != EXTERNAL_STRUCTURED_RECEIPT_SCHEMA or external_structured.get(
        "content_sha256"
    ) != canonical_sha256(external_structured_body):
        raise CoreCreativeDiscoveryError("core external structured benchmark seal changed")
    if root is not None:
        _, _, stored_external_structured = load_stored_pack(root)
        if external_structured != stored_external_structured:
            raise CoreCreativeDiscoveryError("core external structured benchmark component changed")
    validate_component_knockout_preflight(
        value.get("component_knockout_preflight", {}), root or Path.cwd()
    )
    prior_art_portfolio_preflight = value.get("claim_specific_prior_art_portfolio_preflight", {})
    prior_art_portfolio = value.get("claim_specific_prior_art_portfolio", {})
    validate_prior_art_portfolio_preflight(prior_art_portfolio_preflight, root)
    validate_prior_art_portfolio_receipt(prior_art_portfolio, prior_art_portfolio_preflight, root)
    discovery = value.get("discovery_runtime", {})
    if (
        discovery.get("claim_specific_prior_art_automated_screens_complete") is not True
        or discovery.get("claim_specific_prior_art_claims") != 24
        or discovery.get("claim_specific_prior_art_external_request_budget") != 88
        or discovery.get("claim_specific_prior_art_named_human_reviews_complete") is not False
        or discovery.get("claim_specific_prior_art_no_exact_behavior_match_claims") != 16
        or discovery.get("claim_specific_prior_art_status") != "BLOCKED_NAMED_HUMAN_REVIEW_REQUIRED"
        or discovery.get("component_knockout_experiments_preflighted") != 4
        or discovery.get("component_knockout_live_runs_complete") is not False
        or discovery.get("component_knockout_preflight_status")
        != "PASS_PREFLIGHT_LIVE_EXECUTION_NOT_RUN"
        or discovery.get("component_knockout_scheduled_slots") != 384
        or discovery.get("dataset_challenge_kinds")
        != ["intervention", "noisy", "shifted", "unidentifiable"]
        or discovery.get("dataset_positive_controls_passed") != 4
        or discovery.get("dataset_mutation_controls_rejected") != 4
        or discovery.get("dataset_challenge_status") != "PASS_EXECUTABLE_DATASET_CHALLENGES"
        or discovery.get("external_dataset_challenge_kinds")
        != ["intervention", "noisy", "shifted", "unidentifiable"]
        or discovery.get("external_dataset_challenges_passed") != 4
        or discovery.get("external_dataset_mutation_controls_rejected") != 4
        or discovery.get("external_dataset_status") != "PASS_EXTERNAL_DATASET_CHALLENGES"
        or discovery.get("external_structured_benchmark_families")
        != ["tensor_identity", "transform_relation", "variational_functional"]
        or discovery.get("external_structured_benchmark_level5_eligible") is not False
        or discovery.get("external_structured_benchmark_status")
        != "CREATIVITY_BENCHMARK_READY_LEVEL5_BLOCKED_UNSIGNED_SOURCE"
        or discovery.get("external_structured_benchmark_tasks") != 12
        or discovery.get("first_principles_d4_basis_collapse_mutations_rejected") != 5
        or discovery.get("first_principles_d4_controls_passed") != 5
        or discovery.get("first_principles_d4_dimension_mutations_rejected") != 6
        or discovery.get("first_principles_d4_invariant_coordinates") != 6
        or discovery.get("first_principles_d4_multi_coordinate_controls") != 1
        or discovery.get("first_principles_d4_status")
        != "PASS_SYMMETRY_DIMENSION_MULTI_COORDINATE_DERIVATION"
        or discovery.get("first_principles_d4_symmetry_mutations_rejected") != 5
        or discovery.get("learned_invariant_deployment_repaired_coordinates") != 6
        or discovery.get("learned_invariant_identified_passes") != 1
        or discovery.get("learned_invariant_problems") != 3
        or discovery.get("learned_invariant_shift_rejections") != 1
        or discovery.get("learned_invariant_status") != "PASS_LEARNED_MULTI_INVARIANT_CONTROLS"
        or discovery.get("learned_invariant_training_coordinates_retained") != 7
        or discovery.get("learned_invariant_underdetermined_controls") != 1
        or discovery.get("level5_admission_status")
        != level5_admission.get("summary", {}).get("status")
        or discovery.get("level5_campaign_local_process_passes")
        != level5_admission.get("summary", {}).get("campaign_local_level5_process_passes")
        or discovery.get("level5_process_passes_before_external_signature")
        != level5_admission.get("summary", {}).get(
            "process_passes_before_external_signature"
        )
        or discovery.get("state_pair_algebraically_independent_coordinates") != 8
        or discovery.get("state_pair_controls") != 7
        or discovery.get("state_pair_deployment_failures") != 0
        or discovery.get("state_pair_feature_grammar_kinds")
        != ["laurent_monomials", "logarithmic_coordinates", "polynomial_monomials"]
        or discovery.get("state_pair_higher_degree_controls") != 1
        or discovery.get("state_pair_matrix_action_controls") != 2
        or discovery.get("state_pair_multivariate_rational_action_controls") != 1
        or discovery.get("state_pair_nonlinear_action_controls") != 2
        or discovery.get("state_pair_rational_action_controls") != 2
        or discovery.get("state_pair_status") != "PASS_EXACT_TYPED_STATE_PAIR_INVARIANT_CONTROLS"
        or discovery.get("state_pair_target_blind_controls") != 7
        or discovery.get("state_pair_transcendental_action_controls") != 1
        or discovery.get("uncertain_invariant_censored_controls") != 1
        or discovery.get("uncertain_invariant_controls") != 5
        or discovery.get("uncertain_invariant_dependent_joint_controls") != 1
        or discovery.get("uncertain_invariant_deployment_failed_candidates") != 4
        or discovery.get("uncertain_invariant_deployment_surviving_candidates") != 5
        or discovery.get("uncertain_invariant_missingness_controls") != 1
        or discovery.get("uncertain_invariant_marginal_false_positives_rejected") != 3
        or discovery.get("uncertain_invariant_noisy_controls") != 1
        or discovery.get("uncertain_invariant_status")
        != "PASS_COUPLED_UNCERTAIN_INVARIANT_BRANCH_CONTROLS"
        or discovery.get("uncertain_invariant_training_candidates_retained") != 9
        or discovery.get("uncertain_invariant_unit_hypothesis_branches_retained") != 2
        or discovery.get("uncertain_invariant_unit_uncertainty_controls") != 1
        or discovery.get("independent_proof_plan_mechanisms")
        != [
            "induction",
            "invariant_preservation",
            "bijection_or_involution",
            "minimal_counterexample_descent",
            "transform_and_extract",
            "contradiction",
        ]
        or discovery.get("independent_proof_plan_routes_closed") != 6
        or discovery.get("independent_proof_plan_mutations_rejected") != 6
        or discovery.get("independent_proof_plan_status") != "PASS_INDEPENDENT_PROOF_PLAN_SEARCH"
        or discovery.get("retained_piecewise_admitted") != 8
        or discovery.get("retained_piecewise_exact_agreements") != 8
        or discovery.get("retained_piecewise_origin_counts")
        != {"cross_domain_synthesis": 1, "uncertain": 7}
        or discovery.get("retained_piecewise_resource_matched_controls") != 8
        or discovery.get("retained_piecewise_replay_status") != "PASS_RETAINED_PIECEWISE_REPLAY"
        or discovery.get("retained_piecewise_train_exact_holdout_failed") != 1
        or discovery.get("retained_piecewise_zero_holdout_bounded_unknown") != 0
        or discovery.get("retained_piecewise_descendant_admitted") != 16
        or discovery.get("retained_piecewise_descendant_calls") != 6
        or discovery.get("retained_piecewise_descendant_exact_agreements") != 16
        or discovery.get("retained_piecewise_descendant_ideas") != 24
        or discovery.get("retained_piecewise_descendant_nonexecutable_retained") != 8
        or discovery.get("retained_piecewise_descendant_parent_branches_preserved") != 8
        or discovery.get("retained_piecewise_descendant_resource_matched_controls") != 16
        or discovery.get("retained_piecewise_descendant_status")
        != "PASS_LIVE_RETAINED_PIECEWISE_DESCENDANT_CAMPAIGN"
        or discovery.get("retained_piecewise_descendant_zero_holdout_bounded_unknown") != 0
        or discovery.get("retained_piecewise_descendant_zero_holdout_known_control") != 6
        or discovery.get("typed_formula_kinds")
        != [
            "finite_product",
            "finite_sum",
            "generating_function",
            "modular_relation",
            "piecewise_relation",
            "recurrence",
            "tensor_identity",
            "variational_functional",
        ]
        or discovery.get("typed_grammar_controls_passed") != 8
        or discovery.get("typed_grammar_status") != "PASS_EXPANDED_TYPED_GRAMMAR_CONTROLS"
    ):
        raise CoreCreativeDiscoveryError("core expanded typed grammar evidence changed")
    verification = value.get("verification", {})
    if (
        verification.get("multi_host_status") != "PASS_MULTI_HOST_CORE_LLM_EVIDENCE_REPRODUCTION"
        or verification.get("received_machines", 0) < 2
        or verification.get("lean_kernel_checked") is not True
        or verification.get("serious_claim_ladder_status")
        != "PASS_CANDIDATE_BOUND_LADDER_CALIBRATION"
        or verification.get("serious_claim_backend_mutations_rejected") != 10
        or verification.get("serious_claim_lean_mutation_artifact_bound") is not True
        or tuple(verification.get("serious_claim_required_stage_order", ())) != SERIOUS_CLAIM_STAGES
        or verification.get("serious_claims_released_by_ladder") != 0
    ):
        raise CoreCreativeDiscoveryError("core verification evidence changed")
    claims = value.get("claims", {})
    if any(claims.get(key) is not False for key in claims):
        raise CoreCreativeDiscoveryError("core claim boundary changed")
    if value.get("release_gate", {}).get("component_knockout_live_runs_complete") is not False:
        raise CoreCreativeDiscoveryError("core component-knockout release boundary changed")
    if (
        value.get("release_gate", {}).get("level5_process_passes")
        != level5_admission.get("summary", {}).get(
            "admitted_independently_reproduced_level5_successes"
        )
        or value.get("release_gate", {}).get("minimum_level5_process_passes")
        != level5_admission.get("summary", {}).get("minimum_required_before_open_problem")
        or value.get("release_gate", {}).get("open_problem_authorized")
        is not level5_admission.get("release_gate", {}).get("open_problem_authorized")
        or value.get("release_gate", {}).get("status")
        != (
            "READY_FAMOUS_OPEN_PROBLEM_PREREGISTRATION"
            if level5_admission.get("release_gate", {}).get("open_problem_authorized") is True
            else "BLOCKED_CALIBRATION_OR_HUMAN_REVIEW_INCOMPLETE"
        )
    ):
        raise CoreCreativeDiscoveryError("core level-5 admission release boundary changed")
    if (
        value.get("release_gate", {}).get("claim_specific_prior_art_automated_screens_complete")
        is not True
        or value.get("release_gate", {}).get(
            "claim_specific_prior_art_named_human_reviews_complete"
        )
        is not False
        or value.get("release_gate", {}).get("human_prior_art_reviews_complete") is not False
    ):
        raise CoreCreativeDiscoveryError("core prior-art portfolio release boundary changed")
    if (
        value.get("release_gate", {}).get("retained_piecewise_descendant_live_run_complete")
        is not True
    ):
        raise CoreCreativeDiscoveryError("core descendant live-run release boundary changed")
    if root is not None:
        root = root.resolve()
        bindings = value.get("source_bindings", {})
        config_binding = bindings.get("config", {})
        if config_binding.get("path") != CONFIG_PATH or config_binding.get(
            "sha256"
        ) != _normalized_file_sha256(root / CONFIG_PATH):
            raise CoreCreativeDiscoveryError("core config source binding changed")
        context_source_binding = bindings.get("prompt_context_source", {})
        if context_source_binding.get(
            "path"
        ) != PROMPT_CONTEXT_SOURCE_PATH or context_source_binding.get(
            "sha256"
        ) != _normalized_file_sha256(root / PROMPT_CONTEXT_SOURCE_PATH):
            raise CoreCreativeDiscoveryError("core prompt context source binding changed")
        for key, expected_path in (
            ("core_application_source", SOURCE_PATH),
            ("claude_api_source", CLAUDE_API_SOURCE_PATH),
            ("external_campaign_source", EXTERNAL_CAMPAIGN_SOURCE_PATH),
        ):
            source_binding = bindings.get(key, {})
            if source_binding.get("path") != expected_path or source_binding.get(
                "sha256"
            ) != _normalized_file_sha256(root / expected_path):
                raise CoreCreativeDiscoveryError(f"core executable source binding changed: {key}")
        for key in (
            "claim_specific_prior_art_portfolio_preflight_receipt",
            "claim_specific_prior_art_portfolio_receipt",
            "component_knockout_preflight_receipt",
            "dataset_challenge_receipt",
            "declarative_operational_receipt",
            "external_dataset_challenge_receipt",
            "external_structured_benchmark_receipt",
            "expanded_typed_grammar_receipt",
            "learned_invariant_discovery_receipt",
            "level5_success_admission_receipt",
            "multi_host_reproduction_receipt",
            "proof_plan_search_receipt",
            "retained_piecewise_descendant_campaign_receipt",
            "retained_piecewise_replay_receipt",
            "serious_claim_verification_ladder_receipt",
            "state_pair_invariant_discovery_receipt",
            "symmetry_dimension_derivation_receipt",
            "uncertain_invariant_discovery_receipt",
        ):
            binding = bindings.get(key, {})
            path = (root / str(binding.get("path", ""))).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise CoreCreativeDiscoveryError("core receipt binding escapes root") from error
            bound = json.loads(path.read_text(encoding="utf-8"))
            if binding.get("content_sha256") != bound.get("content_sha256"):
                raise CoreCreativeDiscoveryError("core receipt source binding changed")
            if key == "expanded_typed_grammar_receipt":
                validate_expanded_grammar_receipt(bound, root)
            if key == "dataset_challenge_receipt":
                validate_dataset_challenges(bound, root)
            if key == "external_dataset_challenge_receipt":
                validate_external_dataset_challenges(bound, root)
            if key == "external_structured_benchmark_receipt":
                _, _, structured_receipt = load_stored_pack(root, binding["path"])
                if bound != structured_receipt:
                    raise CoreCreativeDiscoveryError(
                        "core structured benchmark packet binding changed"
                    )
            if key == "proof_plan_search_receipt":
                validate_proof_plan_search(bound, root)
            if key == "retained_piecewise_replay_receipt":
                validate_piecewise_replay(bound, root)
            if key == "retained_piecewise_descendant_campaign_receipt":
                validate_piecewise_descendant_campaign(bound, root)
            if key == "serious_claim_verification_ladder_receipt":
                validate_serious_claim_ladder(bound, root)
            if key == "symmetry_dimension_derivation_receipt":
                validate_symmetry_dimension_derivation(bound, root)
            if key == "learned_invariant_discovery_receipt":
                validate_learned_invariants(bound, root)
            if key == "level5_success_admission_receipt":
                validate_level5_success_admission(bound, root)
            if key == "state_pair_invariant_discovery_receipt":
                validate_state_pair_invariants(bound, root)
            if key == "uncertain_invariant_discovery_receipt":
                validate_uncertain_invariants(bound, root)
            if key == "component_knockout_preflight_receipt":
                validate_component_knockout_preflight(bound, root)
            if key == "claim_specific_prior_art_portfolio_preflight_receipt":
                validate_prior_art_portfolio_preflight(bound, root)
                if bound != prior_art_portfolio_preflight:
                    raise CoreCreativeDiscoveryError(
                        "core prior-art portfolio preflight component changed"
                    )
            if key == "claim_specific_prior_art_portfolio_receipt":
                validate_prior_art_portfolio_receipt(bound, prior_art_portfolio_preflight, root)
                if bound != prior_art_portfolio:
                    raise CoreCreativeDiscoveryError("core prior-art portfolio component changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run live Claude-backed core discovery")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--credential-file", type=Path)
    run.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    health = subparsers.add_parser(
        "health", help="run one authenticated, context-bound core LLM health call"
    )
    health.add_argument("--root", type=Path, default=Path.cwd())
    health.add_argument("--credential-file", type=Path)
    health.add_argument(
        "--output",
        type=Path,
        default=Path("runs/math/core-creative-discovery/live-llm-health.json"),
    )
    validate_health = subparsers.add_parser(
        "validate-health", help="validate a sanitized core LLM health receipt offline"
    )
    validate_health.add_argument("--root", type=Path, default=Path.cwd())
    validate_health.add_argument(
        "--receipt",
        type=Path,
        default=Path("runs/math/core-creative-discovery/live-llm-health.json"),
    )
    rebind_health = subparsers.add_parser(
        "rebind-health",
        help="preserve live health-call evidence while rebinding deterministic sources",
    )
    rebind_health.add_argument("--root", type=Path, default=Path.cwd())
    rebind_health.add_argument(
        "--previous",
        type=Path,
        default=Path("runs/math/core-creative-discovery/live-llm-health.json"),
    )
    rebind_health.add_argument(
        "--output",
        type=Path,
        default=Path("runs/math/core-creative-discovery/live-llm-health.json"),
    )
    rebind = subparsers.add_parser(
        "rebind", help="preserve sanitized LLM evidence while rebinding deterministic gates"
    )
    rebind.add_argument("--root", type=Path, default=Path.cwd())
    rebind.add_argument("--previous", type=Path, default=Path(OUTPUT_PATH))
    rebind.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate", help="validate a sanitized core receipt")
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    validate.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "health":
        from .core_llm_health import run_live_health

        health_receipt = run_live_health(args.root, credential_file=args.credential_file)
        output = args.output if args.output.is_absolute() else args.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(health_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "api_response_id": health_receipt["call"]["api_response_id"],
                    "content_sha256": health_receipt["content_sha256"],
                    "credential_source": health_receipt["credential_activation"]["source_kind"],
                    "status": health_receipt["release_gate"]["status"],
                    "usage": health_receipt["call"]["usage"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-health":
        from .core_llm_health import validate_health_receipt

        receipt_path = args.receipt if args.receipt.is_absolute() else args.root / args.receipt
        health_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_health_receipt(health_receipt, args.root)
        print(
            json.dumps(
                {
                    "api_response_id": health_receipt["call"]["api_response_id"],
                    "content_sha256": health_receipt["content_sha256"],
                    "status": health_receipt["release_gate"]["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "rebind-health":
        from .core_llm_health import rebind_health_receipt

        previous_path = args.previous if args.previous.is_absolute() else args.root / args.previous
        health_receipt = rebind_health_receipt(
            args.root, json.loads(previous_path.read_text(encoding="utf-8"))
        )
        output = args.output if args.output.is_absolute() else args.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(health_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "api_response_id": health_receipt["call"]["api_response_id"],
                    "content_sha256": health_receipt["content_sha256"],
                    "status": health_receipt["release_gate"]["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_receipt(receipt, args.root)
    elif args.command == "rebind":
        previous_path = args.previous if args.previous.is_absolute() else args.root / args.previous
        receipt = rebind_core_receipt(
            args.root, json.loads(previous_path.read_text(encoding="utf-8"))
        )
        output = args.output if args.output.is_absolute() else args.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        receipt = run_core(args.root, credential_file=args.credential_file)
        output = args.output if args.output.is_absolute() else args.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "claude_status": receipt["claude_runtime"]["status"],
                "completed_calls": receipt["claude_runtime"]["completed_calls"],
                "content_sha256": receipt["content_sha256"],
                "release_status": receipt["release_gate"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
