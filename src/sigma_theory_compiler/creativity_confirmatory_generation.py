"""Run a no-replacement, attempt-journaled creativity comparison.

Every arm/role slot is dispatched at most once.  The private hash-chained journal records dispatch
before transport and response/error before parsing.  Contract, API, budget, and indeterminate
failures become scored system outcomes; they are never silently retried or replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import creativity_tournament_generation as pilot
from .claude_creativity_api import (
    ClaudeAPIConfig,
    ClaudeBudget,
    ClaudeCallResult,
    ClaudeCallStatus,
    ClaudeCreativityClient,
    ClaudeCreativityError,
    ClaudeRole,
    ClaudeStructuredOutput,
    Transport,
    urllib_transport,
)
from .core_credential import CredentialActivationError, activated_credential
from .durable_llm_attempt_journal import (
    AttemptJournalError,
    DurableAttemptJournal,
    JournaledScheduledTransport,
)
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/creativity_confirmatory_generation.json"
RUNNER_PATH = "src/sigma_theory_compiler/creativity_confirmatory_generation.py"
JOURNAL_SOURCE_PATH = "src/sigma_theory_compiler/durable_llm_attempt_journal.py"
PILOT_HELPER_PATH = "src/sigma_theory_compiler/creativity_tournament_generation.py"
CLAUDE_ADAPTER_PATH = "src/sigma_theory_compiler/claude_creativity_api.py"
CONFIG_SCHEMA = "invariant-creativity-confirmatory-generation-config-1.0"
REVIEW_SCHEMA = "invariant-creativity-confirmatory-review-packet-1.0"
PUBLIC_RECEIPT_SCHEMA = "invariant-creativity-confirmatory-public-generation-1.0"
COORDINATOR_SCHEMA = "invariant-creativity-confirmatory-private-coordinator-1.0"
_ARMS = ("baseline", "full_creativity_first")
_HEX = frozenset("0123456789abcdef")


class ConfirmatoryGenerationError(ValueError):
    """The confirmatory pairing, attempt audit, blinding, or source binding failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ConfirmatoryGenerationError(f"{label} keys changed")


def _sha(value: Any, label: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise ConfirmatoryGenerationError(f"{label} is not a lowercase digest")
    return value


def _source_bindings(root: Path) -> dict[str, Any]:
    paths = {
        "claude_adapter": CLAUDE_ADAPTER_PATH,
        "confirmatory_runner": RUNNER_PATH,
        "durable_attempt_journal": JOURNAL_SOURCE_PATH,
        "pilot_helper": PILOT_HELPER_PATH,
        "config": CONFIG_PATH,
    }
    return {
        name: {"path": path, "sha256": pilot._normalized_file_sha256(root / path)}
        for name, path in sorted(paths.items())
    }


def load_config(root: Path) -> dict[str, Any]:
    value = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    _strict(
        value,
        {
            "attempt_policy",
            "baseline_commit",
            "claude",
            "execution",
            "experiment_id",
            "generation_packet",
            "matched_resource_budget",
            "prompt_policies",
            "review",
            "schema_version",
            "treatment_commit",
        },
        "confirmatory config",
    )
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["experiment_id"]
        != "creativity-first-vs-falsification-first-confirmatory-001"
    ):
        raise ConfirmatoryGenerationError("confirmatory config identity changed")
    for key in ("baseline_commit", "treatment_commit"):
        _sha(value[key], key, length=40)
    packet = value["generation_packet"]
    _strict(packet, {"content_sha256", "path", "required_tasks"}, "generation binding")
    _sha(packet["content_sha256"], "generation content hash")
    if packet["required_tasks"] != 24 or "target" in packet["path"].lower():
        raise ConfirmatoryGenerationError("confirmatory generation binding changed")
    claude = value["claude"]
    _strict(
        claude,
        {
            "credential_env_var",
            "effort",
            "maximum_output_tokens_per_call",
            "maximum_scheduled_calls_per_arm",
            "maximum_total_tokens_per_arm",
            "model",
            "timeout_seconds",
        },
        "confirmatory Claude policy",
    )
    if (
        claude["credential_env_var"] != "ANTHROPIC_API_KEY"
        or claude["model"] != "claude-opus-4-6"
        or claude["effort"] != "high"
        or claude["maximum_scheduled_calls_per_arm"] != 48
        or claude["maximum_output_tokens_per_call"] < 4096
    ):
        raise ConfirmatoryGenerationError("confirmatory Claude policy weakened")
    attempts = value["attempt_policy"]
    _strict(
        attempts,
        {
            "attempt_journal_required",
            "contract_failure_policy",
            "persist_dispatch_before_transport",
            "persist_response_before_validation",
            "replacement_calls_allowed",
            "transient_retries_allowed",
        },
        "confirmatory attempt policy",
    )
    if attempts != {
        "attempt_journal_required": True,
        "contract_failure_policy": "retain_admitted_content_and_count_failure",
        "persist_dispatch_before_transport": True,
        "persist_response_before_validation": True,
        "replacement_calls_allowed": False,
        "transient_retries_allowed": False,
    }:
        raise ConfirmatoryGenerationError("confirmatory no-replacement policy changed")
    resource = value["matched_resource_budget"]
    if (
        resource.get("calls_per_task") != 2
        or resource.get("tokens_per_arm") != claude["maximum_total_tokens_per_arm"]
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in resource.values())
    ):
        raise ConfirmatoryGenerationError("confirmatory resources are not matched")
    if set(value["prompt_policies"]) != set(_ARMS):
        raise ConfirmatoryGenerationError("confirmatory prompt arms changed")
    for arm in _ARMS:
        _strict(
            value["prompt_policies"][arm],
            {"critic_instruction", "proposer_instruction", "system_instruction"},
            "confirmatory prompt policy",
        )
    execution = value["execution"]
    if (
        execution.get("requested_hypotheses_per_task") != 3
        or execution.get("maximum_hypotheses_per_task", 99) > 8
        or not str(execution.get("arm_order_seed", "")).startswith("sha256:")
    ):
        raise ConfirmatoryGenerationError("confirmatory execution policy changed")
    review = value["review"]
    if (
        review.get("minimum_named_reviewers", 0) < 2
        or review.get("axes") != ["coherence", "nontriviality", "followup_value"]
        or review.get("useful_threshold_each_axis") != 3
    ):
        raise ConfirmatoryGenerationError("confirmatory review policy weakened")
    return value


def _load_generation(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["generation_packet"]
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ConfirmatoryGenerationError("confirmatory generation packet escapes repository") from error
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version")
        != "invariant-rotating-external-generation-packet-1.0"
        or value.get("content_sha256") != binding["content_sha256"]
        or len(value.get("tasks", [])) != binding["required_tasks"]
    ):
        raise ConfirmatoryGenerationError("confirmatory generation packet binding changed")
    serialized = json.dumps(value, sort_keys=True).lower()
    if '"holdout"' in serialized or '"source_uri"' in serialized or '"source_id"' in serialized:
        raise ConfirmatoryGenerationError("confirmatory generation packet leaked sealed material")
    return value


def _client_config(config: Mapping[str, Any]) -> ClaudeAPIConfig:
    claude = config["claude"]
    return ClaudeAPIConfig(
        model=claude["model"],
        credential_env_var=claude["credential_env_var"],
        execution_enabled=True,
        maximum_calls=64,
        maximum_total_tokens=2_000_000,
        maximum_output_tokens=claude["maximum_output_tokens_per_call"],
        timeout_seconds=claude["timeout_seconds"],
        effort=claude["effort"],
    )


def _call_id(task_id: str, arm: str, role: ClaudeRole) -> str:
    return f"{task_id}:{arm}:{role.value}"


def _outcome_event(journal: DurableAttemptJournal, call_id: str) -> dict[str, Any] | None:
    events = [
        item for item in journal.events_for(call_id) if item["event_kind"] == "scheduled_call_outcome"
    ]
    if len(events) > 1:
        raise ConfirmatoryGenerationError("scheduled call has multiple outcomes")
    return None if not events else dict(events[0]["payload"])


def _result_from_dict(value: Mapping[str, Any] | None) -> ClaudeCallResult | None:
    if value is None:
        return None
    output_raw = value.get("output")
    output = None
    if output_raw is not None:
        parseable = {
            key: item for key, item in output_raw.items() if key != "quarantine"
        }
        output = ClaudeStructuredOutput.from_mapping(parseable)
    return ClaudeCallResult(
        ClaudeCallStatus(value["status"]),
        ClaudeRole(value["role"]),
        value["benchmark_id"],
        output,
        dict(value["evidence"]),
    )


def _response_usage(journal: DurableAttemptJournal, call_id: str) -> tuple[int, int]:
    responses = [
        item for item in journal.events_for(call_id) if item["event_kind"] == "message_response"
    ]
    if len(responses) > 1:
        raise ConfirmatoryGenerationError("scheduled call has multiple provider responses")
    if not responses:
        return 0, 0
    usage = responses[0]["payload"].get("response", {}).get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (input_tokens, output_tokens)):
        raise ConfirmatoryGenerationError("journaled provider usage is invalid")
    return input_tokens, output_tokens


def _arm_usage(journal: DurableAttemptJournal, arm: str) -> int:
    total = 0
    for event in journal.events:
        if event["event_kind"] == "message_response" and event["payload"].get("arm") == arm:
            usage = event["payload"].get("response", {}).get("usage", {})
            total += int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    return total


def _restore_client_budgets(
    clients: Mapping[str, ClaudeCreativityClient], journal: DurableAttemptJournal
) -> None:
    for arm, client in clients.items():
        calls = []
        for event in journal.events:
            if (
                event["event_kind"] == "scheduled_call_outcome"
                and event["payload"].get("arm") == arm
                and event["payload"].get("result") is not None
            ):
                result = event["payload"]["result"]
                if result.get("status") == "completed":
                    calls.append(result)
        input_tokens = sum(item["evidence"]["usage"]["input_tokens"] for item in calls)
        output_tokens = sum(item["evidence"]["usage"]["output_tokens"] for item in calls)
        client.budget = ClaudeBudget(len(calls), input_tokens, output_tokens)


def _contract_errors(
    result: ClaudeCallResult,
    role: ClaudeRole,
    config: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors = []
    if result.status is not ClaudeCallStatus.COMPLETED:
        errors.append(f"client_status:{result.status.value}")
        return errors
    try:
        if role is ClaudeRole.PROPOSER:
            pilot._validate_proposer(result, config)
        else:
            pilot._validate_critic(result, summaries)
    except pilot.TournamentGenerationError as error:
        errors.append(str(error))
    return errors


def _append_outcome(
    journal: DurableAttemptJournal,
    call_id: str,
    *,
    arm: str,
    task_id: str,
    role: ClaudeRole,
    status: str,
    result: ClaudeCallResult | None,
    contract_errors: Sequence[str],
) -> dict[str, Any]:
    event = journal.append(
        "scheduled_call_outcome",
        call_id,
        {
            "arm": arm,
            "contract_errors": list(contract_errors),
            "result": None if result is None else result.to_dict(),
            "role": role.value,
            "status": status,
            "task_id": task_id,
        },
    )
    return dict(event["payload"])


def _run_scheduled(
    journal: DurableAttemptJournal,
    client: ClaudeCreativityClient,
    *,
    arm: str,
    task_id: str,
    role: ClaudeRole,
    public_task: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    base_transport: Transport,
) -> tuple[dict[str, Any], ClaudeCallResult | None]:
    call_id = _call_id(task_id, arm, role)
    existing = _outcome_event(journal, call_id)
    if existing is not None:
        return existing, _result_from_dict(existing.get("result"))
    previous = journal.events_for(call_id)
    if any(item["event_kind"] == "message_dispatch" for item in previous):
        outcome = _append_outcome(
            journal,
            call_id,
            arm=arm,
            task_id=task_id,
            role=role,
            status="indeterminate_after_dispatch",
            result=None,
            contract_errors=["process_stopped_after_dispatch_before_durable_outcome"],
        )
        return outcome, None
    if _arm_usage(journal, arm) >= config["claude"]["maximum_total_tokens_per_arm"]:
        outcome = _append_outcome(
            journal,
            call_id,
            arm=arm,
            task_id=task_id,
            role=role,
            status="budget_blocked",
            result=None,
            contract_errors=["arm_token_budget_exhausted_before_dispatch"],
        )
        return outcome, None
    client.transport = JournaledScheduledTransport(
        journal,
        scheduled_call_id=call_id,
        arm=arm,
        task_id=task_id,
        role=role.value,
        base_transport=base_transport,
    )
    policy = config["prompt_policies"][arm]
    try:
        result = client.run(
            role,
            task_id,
            public_task,
            candidate_summaries=summaries,
            instruction_override=policy[f"{role.value}_instruction"],
            system_override=policy["system_instruction"],
            hypothesis_slots=(
                config["execution"]["requested_hypotheses_per_task"]
                if role is ClaudeRole.PROPOSER
                else None
            ),
        )
    except (ClaudeCreativityError, AttemptJournalError, OSError, TimeoutError) as error:
        outcome = _append_outcome(
            journal,
            call_id,
            arm=arm,
            task_id=task_id,
            role=role,
            status="client_or_transport_failure",
            result=None,
            contract_errors=[f"{type(error).__name__}:{str(error)[:1024]}"],
        )
        return outcome, None
    errors = _contract_errors(result, role, config, summaries)
    outcome = _append_outcome(
        journal,
        call_id,
        arm=arm,
        task_id=task_id,
        role=role,
        status="contract_pass" if not errors else "contract_failure",
        result=result,
        contract_errors=errors,
    )
    return outcome, result


def _sentinel_summary(task_id: str, arm: str) -> list[dict[str, Any]]:
    return [
        {
            "behavior_sha256": canonical_sha256(
                {"arm": arm, "status": "proposer_contract_failure", "task": task_id}
            ),
            "candidate_id": "candidate.failure."
            + canonical_sha256({"arm": arm, "task": task_id})[:20],
            "expression": "NO_SCHEMA_ADMITTED_HYPOTHESIS",
            "representation": "other_typed_relation",
            "typed_status": "PROPOSER_CONTRACT_FAILURE_RETAIN_FOR_REPAIR",
        }
    ]


def _empty_critic(task_id: str) -> ClaudeCallResult:
    output = ClaudeStructuredOutput(ClaudeRole.CRITIC, task_id, (), ())
    return ClaudeCallResult(
        ClaudeCallStatus.COMPLETED,
        ClaudeRole.CRITIC,
        task_id,
        output,
        {"synthetic_for_branch_construction": True},
    )


def _failure_branch(
    task_id: str, proposer_outcome: Mapping[str, Any], critic_outcome: Mapping[str, Any]
) -> dict[str, Any]:
    branch = {
        "behavior_sha256": canonical_sha256(
            {"status": "no_admitted_hypothesis", "task": task_id}
        ),
        "branch_kind": "scheduled_generation_contract_failure",
        "expression": "No schema-admitted hypothesis was produced in the scheduled proposer slot.",
        "family": "contract_failure",
        "falsifiers": ["a schema-admitted hypothesis exists in the private raw response"],
        "generation_contract_status": "failed",
        "initial_check_status": "failed",
        "invariants": ["scheduled_slot_was_not_replaced"],
        "known_analogues": [],
        "later_used_as_parent": False,
        "llm_origin_assessment": "uncertain",
        "proof_mechanism": "contract_repair_required",
        "proof_mechanism_sha256": canonical_sha256("contract_repair_required"),
        "proof_plan": [
            "inspect the private bounded raw response after review",
            "repair the wire or local admission contract without replacing this outcome",
        ],
        "rationale": "A failed scheduled system outcome is retained and scored rather than retried away.",
        "representation": "other_typed_relation",
        "source_domains": ["system_contract"],
        "synthesis_note": "This branch records failure and makes the run ineligible for confirmation.",
        "scheduled_outcomes": [proposer_outcome["status"], critic_outcome["status"]],
    }
    branch["branch_id"] = "branch." + canonical_sha256(branch)[:24]
    return branch


def _build_branches(
    task_id: str,
    arm: str,
    proposer: ClaudeCallResult | None,
    critic: ClaudeCallResult | None,
    proposer_outcome: Mapping[str, Any],
    critic_outcome: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if proposer is None or proposer.output is None or not proposer.output.hypotheses:
        return [_failure_branch(task_id, proposer_outcome, critic_outcome)]
    branches = pilot._hypothesis_branches(
        proposer, critic or _empty_critic(task_id), arm, task_id, config
    )
    status = (
        "pass"
        if proposer_outcome["status"] == "contract_pass"
        and critic_outcome["status"] == "contract_pass"
        else "failed"
    )
    for branch in branches:
        branch["generation_contract_status"] = status
        branch["scheduled_outcomes"] = [
            proposer_outcome["status"],
            critic_outcome["status"],
        ]
        branch["branch_id"] = "branch." + canonical_sha256(
            {key: value for key, value in branch.items() if key != "branch_id"}
        )[:24]
    if arm == "full_creativity_first":
        branches.extend(
            pilot._recombination_branches(
                branches,
                task_id,
                config["execution"]["treatment_recombinations_per_task"],
            )
        )
        for branch in branches:
            branch.setdefault("generation_contract_status", status)
            branch.setdefault(
                "scheduled_outcomes",
                [proposer_outcome["status"], critic_outcome["status"]],
            )
            branch["branch_id"] = "branch." + canonical_sha256(
                {key: value for key, value in branch.items() if key != "branch_id"}
            )[:24]
    return branches


def _recursive_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden in value or any(_recursive_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_recursive_key(item, forbidden) for item in value)
    return False


def run_generation(
    root: Path,
    *,
    journal: DurableAttemptJournal,
    credential_file: Path | None = None,
    transport: Transport = urllib_transport,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    config = load_config(root)
    generation = _load_generation(root, config)
    if journal.header.get("experiment_id") != config["experiment_id"]:
        raise ConfirmatoryGenerationError("attempt journal experiment binding changed")
    if journal.header.get("source_bindings") != _source_bindings(root):
        raise ConfirmatoryGenerationError("attempt journal source binding changed")
    clients = {
        arm: ClaudeCreativityClient(_client_config(config), transport) for arm in _ARMS
    }
    _restore_client_budgets(clients, journal)
    environment = None
    if credential_file is not None:
        environment = dict(os.environ)
        environment[config["claude"]["credential_env_var"]] = ""
        environment["INVARIANT_ENV_FILE"] = str(credential_file.resolve())
    task_results: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        with activated_credential(
            project_root=root,
            env_var=config["claude"]["credential_env_var"],
            environment=environment,
        ) as activation:
            for task in generation["tasks"]:
                task_id = task["task_id"]
                public = pilot._public_task(task)
                task_results[task_id] = {}
                proposer_results: dict[str, ClaudeCallResult | None] = {}
                proposer_outcomes: dict[str, dict[str, Any]] = {}
                summaries: dict[str, list[dict[str, Any]]] = {}
                for arm in pilot._arm_order(config, task_id, "proposer"):
                    outcome, result = _run_scheduled(
                        journal,
                        clients[arm],
                        arm=arm,
                        task_id=task_id,
                        role=ClaudeRole.PROPOSER,
                        public_task=public,
                        summaries=(),
                        config=config,
                        base_transport=transport,
                    )
                    proposer_outcomes[arm] = outcome
                    proposer_results[arm] = result
                    summaries[arm] = (
                        pilot._candidate_summaries(result, arm)
                        if result is not None
                        and result.output is not None
                        and result.output.hypotheses
                        else _sentinel_summary(task_id, arm)
                    )
                for arm in pilot._arm_order(config, task_id, "critic"):
                    critic_outcome, critic = _run_scheduled(
                        journal,
                        clients[arm],
                        arm=arm,
                        task_id=task_id,
                        role=ClaudeRole.CRITIC,
                        public_task=public,
                        summaries=summaries[arm],
                        config=config,
                        base_transport=transport,
                    )
                    branches = _build_branches(
                        task_id,
                        arm,
                        proposer_results[arm],
                        critic,
                        proposer_outcomes[arm],
                        critic_outcome,
                        config,
                    )
                    tokens = sum(
                        sum(_response_usage(journal, _call_id(task_id, arm, role)))
                        for role in (ClaudeRole.PROPOSER, ClaudeRole.CRITIC)
                    )
                    task_results[task_id][arm] = {
                        "branches": branches,
                        "tokens_used": tokens,
                    }
    except (CredentialActivationError, AttemptJournalError) as error:
        raise ConfirmatoryGenerationError(str(error)) from error
    call_outcomes = [
        item["payload"]
        for item in journal.events
        if item["event_kind"] == "scheduled_call_outcome"
    ]
    dispatches = [
        item["payload"] for item in journal.events if item["event_kind"] == "message_dispatch"
    ]
    expected_per_arm = config["claude"]["maximum_scheduled_calls_per_arm"]
    if len(call_outcomes) != 2 * expected_per_arm:
        raise ConfirmatoryGenerationError("confirmatory scheduled outcome coverage is incomplete")
    outcome_by_arm = {
        arm: [item for item in call_outcomes if item["arm"] == arm] for arm in _ARMS
    }
    dispatch_by_arm = {
        arm: [item for item in dispatches if item["arm"] == arm] for arm in _ARMS
    }
    if any(len(outcome_by_arm[arm]) != expected_per_arm for arm in _ARMS):
        raise ConfirmatoryGenerationError("confirmatory scheduled arm balance changed")
    confirmatory_eligible = (
        all(item["status"] == "contract_pass" for item in call_outcomes)
        and all(len(dispatch_by_arm[arm]) == expected_per_arm for arm in _ARMS)
        and all(
            _arm_usage(journal, arm) <= config["claude"]["maximum_total_tokens_per_arm"]
            for arm in _ARMS
        )
    )
    mapping = []
    outputs = []
    key = journal.unblinding_key
    resource = config["matched_resource_budget"]
    for task_id in sorted(task_results):
        paired = []
        for arm in _ARMS:
            blinded_id = pilot._blinded_id(key, task_id, arm)
            result = task_results[task_id][arm]
            paired.append(
                {
                    "blinded_output_id": blinded_id,
                    "branches": sorted(result["branches"], key=lambda item: item["branch_id"]),
                    "resource_budget": {
                        "calls": resource["calls_per_task"],
                        "grammar_depth": resource["grammar_depth"],
                        "tokens": resource["tokens_per_arm"],
                        "verifier_invocations": resource["verifier_invocations_per_task"],
                        "wall_clock_milliseconds": resource["wall_clock_milliseconds_per_call"]
                        * resource["calls_per_task"],
                    },
                    "task_id": task_id,
                    "tokens_used": result["tokens_used"],
                    "typed_usable_ideas": len(result["branches"]),
                }
            )
            mapping.append({"arm": arm, "blinded_output_id": blinded_id, "task_id": task_id})
        outputs.extend(sorted(paired, key=lambda item: item["blinded_output_id"]))
    review: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA,
        "experiment_id": config["experiment_id"],
        "review_policy": config["review"],
        "review_instructions": (
            "Score every branch independently without inferring system identity. Contract-failure "
            "branches are real system outcomes. Literature novelty is outside this review."
        ),
        "blinded_outputs": outputs,
        "claims": {
            "arm_identity_disclosed": False,
            "holdout_opened": False,
            "literature_novelty_established": False,
            "more_creative_established": False,
        },
    }
    review["content_sha256"] = canonical_sha256(review)
    status_counts: dict[str, int] = {}
    for item in call_outcomes:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    public: dict[str, Any] = {
        "schema_version": PUBLIC_RECEIPT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "source_bindings": {
            **_source_bindings(root),
            "generation_packet_content_sha256": generation["content_sha256"],
        },
        "credential_activation": activation.to_evidence(),
        "attempt_accounting": {
            "attempt_journal_content_sha256": journal.content_sha256,
            "balanced_provider_message_attempts": len(dispatch_by_arm["baseline"])
            == len(dispatch_by_arm["full_creativity_first"]),
            "contract_outcome_counts": dict(sorted(status_counts.items())),
            "provider_message_attempts": len(dispatches),
            "replacement_calls": 0,
            "scheduled_slots": len(call_outcomes),
            "transient_retries": 0,
        },
        "claude_runtime": {
            "authenticated_messages_api_working": bool(dispatches),
            "effort": config["claude"]["effort"],
            "model": config["claude"]["model"],
            "status": (
                "PASS_CONFIRMATORY_GENERATION"
                if confirmatory_eligible
                else "COUNTED_FAILURES_NOT_CONFIRMATORY"
            ),
            "total_tokens": sum(_arm_usage(journal, arm) for arm in _ARMS),
        },
        "blinding": {
            "review_packet_content_sha256": review["content_sha256"],
            "unblinding_key_sha256": hashlib.sha256(key).hexdigest(),
            "status": "SEALED_UNTIL_TWO_NAMED_REVIEWS",
        },
        "release_gate": {
            "confirmatory_generation_eligible": confirmatory_eligible,
            "named_blinded_reviews_complete": False,
            "status": "BLOCKED_TWO_NAMED_BLINDED_REVIEWS_REQUIRED",
            "tournament_scored": False,
        },
        "claims": {
            "credential_material_persisted": False,
            "holdout_opened": False,
            "literature_novelty_established": False,
            "more_creative_established": False,
        },
    }
    public["content_sha256"] = canonical_sha256(public)
    coordinator: dict[str, Any] = {
        "schema_version": COORDINATOR_SCHEMA,
        "experiment_id": config["experiment_id"],
        "public_receipt_content_sha256": public["content_sha256"],
        "review_packet_content_sha256": review["content_sha256"],
        "attempt_journal_content_sha256": journal.content_sha256,
        "unblinding_key_hex": key.hex(),
        "mapping": sorted(mapping, key=lambda item: (item["task_id"], item["blinded_output_id"])),
        "arm_outcome_status_counts": {
            arm: {
                status: sum(item["status"] == status for item in outcome_by_arm[arm])
                for status in sorted({row["status"] for row in call_outcomes})
            }
            for arm in _ARMS
        },
        "claims": {"safe_to_publish_before_review": False},
    }
    coordinator["content_sha256"] = canonical_sha256(coordinator)
    validate_public(review, public, root)
    validate_coordinator(coordinator, review, public, journal)
    return review, public, coordinator


def validate_public(
    review: Mapping[str, Any], public: Mapping[str, Any], root: Path | None = None
) -> None:
    for value, schema, label in (
        (review, REVIEW_SCHEMA, "confirmatory review packet"),
        (public, PUBLIC_RECEIPT_SCHEMA, "confirmatory public receipt"),
    ):
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("schema_version") != schema or value.get("content_sha256") != canonical_sha256(body):
            raise ConfirmatoryGenerationError(f"{label} identity or seal changed")
    outputs = review.get("blinded_outputs", [])
    if (
        len(outputs) != 48
        or len({item["blinded_output_id"] for item in outputs}) != 48
        or _recursive_key(review, "arm")
    ):
        raise ConfirmatoryGenerationError("confirmatory public review coverage or blinding changed")
    counts: dict[str, int] = {}
    for item in outputs:
        counts[item["task_id"]] = counts.get(item["task_id"], 0) + 1
        if item["tokens_used"] < 0 or not item["branches"]:
            raise ConfirmatoryGenerationError("confirmatory review output is invalid")
    if len(counts) != 24 or set(counts.values()) != {2}:
        raise ConfirmatoryGenerationError("confirmatory paired task coverage changed")
    attempts = public.get("attempt_accounting", {})
    release = public.get("release_gate", {})
    if (
        attempts.get("scheduled_slots") != 96
        or attempts.get("replacement_calls") != 0
        or attempts.get("transient_retries") != 0
        or release.get("named_blinded_reviews_complete") is not False
        or release.get("tournament_scored") is not False
        or any(review.get("claims", {}).values())
        or any(public.get("claims", {}).values())
    ):
        raise ConfirmatoryGenerationError("confirmatory accounting or claim gate changed")
    if root is not None:
        root = root.resolve()
        config = load_config(root)
        generation = _load_generation(root, config)
        expected = {
            **_source_bindings(root),
            "generation_packet_content_sha256": generation["content_sha256"],
        }
        if public.get("source_bindings") != expected:
            raise ConfirmatoryGenerationError("confirmatory public source bindings changed")


def validate_coordinator(
    coordinator: Mapping[str, Any],
    review: Mapping[str, Any],
    public: Mapping[str, Any],
    journal: DurableAttemptJournal,
) -> None:
    body = {key: item for key, item in coordinator.items() if key != "content_sha256"}
    if (
        coordinator.get("schema_version") != COORDINATOR_SCHEMA
        or coordinator.get("content_sha256") != canonical_sha256(body)
        or coordinator.get("public_receipt_content_sha256") != public["content_sha256"]
        or coordinator.get("review_packet_content_sha256") != review["content_sha256"]
        or coordinator.get("attempt_journal_content_sha256") != journal.content_sha256
        or len(coordinator.get("mapping", [])) != 48
        or coordinator.get("claims") != {"safe_to_publish_before_review": False}
    ):
        raise ConfirmatoryGenerationError("confirmatory private coordinator binding changed")
    key = bytes.fromhex(coordinator["unblinding_key_hex"])
    if key != journal.unblinding_key or hashlib.sha256(key).hexdigest() != public["blinding"][
        "unblinding_key_sha256"
    ]:
        raise ConfirmatoryGenerationError("confirmatory unblinding key binding changed")
    for item in coordinator["mapping"]:
        if pilot._blinded_id(key, item["task_id"], item["arm"]) != item["blinded_output_id"]:
            raise ConfirmatoryGenerationError("confirmatory unblinding mapping changed")


def _private_path(root: Path, path: Path) -> Path:
    return pilot._private_output_path(root, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--credential-file", type=Path)
    run.add_argument("--journal", type=Path, required=True)
    run.add_argument("--review-output", type=Path, required=True)
    run.add_argument("--receipt-output", type=Path, required=True)
    run.add_argument("--coordinator-output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--review-packet", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        root = args.root.resolve()
        journal_path = _private_path(root, args.journal)
        coordinator_path = _private_path(root, args.coordinator_output)
        if journal_path == coordinator_path:
            raise ConfirmatoryGenerationError("attempt journal and coordinator paths must differ")
        config = load_config(root)
        if journal_path.exists():
            journal = DurableAttemptJournal.load(journal_path)
        else:
            journal = DurableAttemptJournal.create(
                journal_path,
                experiment_id=config["experiment_id"],
                source_bindings=_source_bindings(root),
                unblinding_key=secrets.token_bytes(32),
            )
        review, receipt, coordinator = run_generation(
            root,
            journal=journal,
            credential_file=args.credential_file,
        )
        for path, value in (
            (args.review_output, review),
            (args.receipt_output, receipt),
            (coordinator_path, coordinator),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        review = json.loads(args.review_packet.read_text(encoding="utf-8"))
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_public(review, receipt, args.root.resolve())
    print(
        json.dumps(
            {
                "confirmatory_generation_eligible": receipt["release_gate"][
                    "confirmatory_generation_eligible"
                ],
                "content_sha256": receipt["content_sha256"],
                "provider_message_attempts": receipt["attempt_accounting"][
                    "provider_message_attempts"
                ],
                "status": receipt["release_gate"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
