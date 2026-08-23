"""Run one explicitly authorized, durable creativity component-knockout experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from . import creativity_confirmatory_generation as confirmatory
from . import creativity_tournament_generation as pilot
from .claude_creativity_api import (
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
from .creativity_component_knockout_preflight import (
    CONFIG_PATH,
    REFERENCE_ARM,
    SUITE_ID,
    _ordered_arms,
    load_config,
)
from .creativity_component_knockout_preflight import (
    OUTPUT_PATH as PREFLIGHT_OUTPUT_PATH,
)
from .creativity_component_knockout_preflight import (
    validate_receipt as validate_preflight,
)
from .durable_llm_attempt_journal import (
    AttemptJournalError,
    DurableAttemptJournal,
    JournaledScheduledTransport,
)
from .sigma_core import canonical_sha256

RUNNER_PATH = "src/sigma_theory_compiler/creativity_component_knockout_generation.py"
CONFIRMATORY_PATH = "src/sigma_theory_compiler/creativity_confirmatory_generation.py"
PILOT_PATH = "src/sigma_theory_compiler/creativity_tournament_generation.py"
JOURNAL_PATH = "src/sigma_theory_compiler/durable_llm_attempt_journal.py"
CLIENT_PATH = "src/sigma_theory_compiler/claude_creativity_api.py"
CREDENTIAL_PATH = "src/sigma_theory_compiler/core_credential.py"
AUTHORIZATION_SCHEMA = "invariant-creativity-component-knockout-authorization-1.0"
REVIEW_SCHEMA = "invariant-creativity-component-knockout-review-packet-1.0"
PUBLIC_RECEIPT_SCHEMA = "invariant-creativity-component-knockout-public-receipt-1.0"
COORDINATOR_SCHEMA = "invariant-creativity-component-knockout-private-coordinator-1.0"
SCOPE = "one_component_knockout_experiment"
CALLS_PER_EXPERIMENT = 96
TOKENS_PER_EXPERIMENT = 400_000
_HEX = frozenset("0123456789abcdef")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")


class ComponentKnockoutGenerationError(ValueError):
    """Authorization, intervention, transport, blinding, or accounting failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ComponentKnockoutGenerationError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ComponentKnockoutGenerationError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComponentKnockoutGenerationError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise ComponentKnockoutGenerationError(f"{label} is not a JSON object")
    return value


def _normalized_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _reseal(value: dict[str, Any]) -> dict[str, Any]:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    return value


def _experiment(config: Mapping[str, Any], experiment_id: str) -> dict[str, Any]:
    rows = [item for item in config["experiments"] if item["experiment_id"] == experiment_id]
    if len(rows) != 1:
        raise ComponentKnockoutGenerationError("component-knockout experiment ID is not registered")
    return dict(rows[0])


def _preflight(root: Path) -> dict[str, Any]:
    value = _read_json(root / PREFLIGHT_OUTPUT_PATH, "component-knockout preflight")
    validate_preflight(value, root)
    return value


def _journal_path_sha256(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def authorization_template(
    root: Path, experiment_id: str, journal_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config, _ = load_config(root)
    _experiment(config, experiment_id)
    preflight = _preflight(root)
    return _reseal(
        {
            "schema_version": AUTHORIZATION_SCHEMA,
            "suite_id": SUITE_ID,
            "authorization_scope": SCOPE,
            "experiment_id": experiment_id,
            "authorized_executions": 1,
            "journal_path_sha256": _journal_path_sha256(journal_path),
            "preflight_content_sha256": preflight["content_sha256"],
            "maximum_provider_calls": CALLS_PER_EXPERIMENT,
            "maximum_total_tokens": TOKENS_PER_EXPERIMENT,
            "paid_execution_authorized": False,
            "authorized_by": "REPLACE_WITH_AUTHORIZING_PERSON",
            "authorized_at_utc": "REPLACE_WITH_UTC_TIMESTAMP",
            "authorization_nonce": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS",
        }
    )


def seal_authorization(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_sha256", None)
    return _reseal(body)


def validate_authorization(
    value: Mapping[str, Any], root: Path, *, require_approved: bool = True
) -> dict[str, Any]:
    root = root.resolve()
    _strict(
        value,
        {
            "authorization_nonce",
            "authorization_scope",
            "authorized_executions",
            "authorized_at_utc",
            "authorized_by",
            "content_sha256",
            "experiment_id",
            "journal_path_sha256",
            "maximum_provider_calls",
            "maximum_total_tokens",
            "paid_execution_authorized",
            "preflight_content_sha256",
            "schema_version",
            "suite_id",
        },
        "component-knockout authorization",
    )
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value["schema_version"] != AUTHORIZATION_SCHEMA
        or value["suite_id"] != SUITE_ID
        or value["authorization_scope"] != SCOPE
        or value["authorized_executions"] != 1
        or value["content_sha256"] != canonical_sha256(body)
        or value["maximum_provider_calls"] != CALLS_PER_EXPERIMENT
        or value["maximum_total_tokens"] != TOKENS_PER_EXPERIMENT
    ):
        raise ComponentKnockoutGenerationError("component-knockout authorization seal changed")
    config, _ = load_config(root)
    experiment = _experiment(config, str(value["experiment_id"]))
    preflight = _preflight(root)
    if value["preflight_content_sha256"] != preflight["content_sha256"]:
        raise ComponentKnockoutGenerationError("authorization does not bind the current preflight")
    principal = value["authorized_by"]
    timestamp = value["authorized_at_utc"]
    nonce = value["authorization_nonce"]
    if (
        not isinstance(principal, str)
        or not principal.strip()
        or principal.startswith("REPLACE_")
        or len(principal.encode("utf-8")) > 256
        or not isinstance(timestamp, str)
        or _UTC.fullmatch(timestamp) is None
    ):
        raise ComponentKnockoutGenerationError("authorization principal or UTC timestamp is invalid")
    try:
        datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ComponentKnockoutGenerationError("authorization UTC timestamp is invalid") from error
    _sha(nonce, "authorization nonce")
    _sha(value["journal_path_sha256"], "authorization journal path binding")
    if not isinstance(value["paid_execution_authorized"], bool):
        raise ComponentKnockoutGenerationError("paid execution authorization is not boolean")
    if require_approved and value["paid_execution_authorized"] is not True:
        raise ComponentKnockoutGenerationError("paid component-knockout execution is not authorized")
    return experiment


def _source_bindings(root: Path) -> dict[str, Any]:
    config, dependencies = load_config(root)
    paths = {
        "claude_client": CLIENT_PATH,
        "component_knockout_config": CONFIG_PATH,
        "confirmatory_generation": CONFIRMATORY_PATH,
        "core_credential": CREDENTIAL_PATH,
        "durable_attempt_journal": JOURNAL_PATH,
        "generation_packet": config["generation_packet"]["path"],
        "knockout_generation": RUNNER_PATH,
        "knockout_preflight": PREFLIGHT_OUTPUT_PATH,
        "tournament_generation": PILOT_PATH,
    }
    bindings = {
        name: {"path": path, "sha256": _normalized_file_sha256(root / path)}
        for name, path in sorted(paths.items())
    }
    bindings["generation_packet"]["content_sha256"] = dependencies["generation"][
        "content_sha256"
    ]
    bindings["knockout_preflight"]["content_sha256"] = _preflight(root)[
        "content_sha256"
    ]
    return bindings


def _runtime_config(
    config: Mapping[str, Any], dependencies: Mapping[str, Any], experiment: Mapping[str, Any]
) -> dict[str, Any]:
    confirmatory_config = dependencies["confirmatory"]
    arms = (experiment["reference_arm"], experiment["knockout_arm"])
    base_policy = confirmatory_config["prompt_policies"][REFERENCE_ARM]
    policies = {}
    for arm in arms:
        delta = config["arms"][arm]["instruction_delta"]
        if arm == REFERENCE_ARM:
            policies[arm] = dict(base_policy)
        else:
            policies[arm] = {
                key: f"{instruction} Intervention for this arm: {delta}"
                for key, instruction in base_policy.items()
            }
    return {
        "claude": dict(confirmatory_config["claude"]),
        "execution": {
            **confirmatory_config["execution"],
            "arm_order_seed": experiment["arm_order_seed"],
        },
        "prompt_policies": policies,
    }


def _intervention_contract_errors(
    result: ClaudeCallResult,
    role: ClaudeRole,
    arm: str,
    config: Mapping[str, Any],
) -> list[str]:
    if role is not ClaudeRole.PROPOSER or result.output is None:
        return []
    admitted = set(config["arms"][arm]["execution_semantics"]["admitted_representations"])
    excluded = sorted(
        {item.representation for item in result.output.hypotheses if item.representation not in admitted}
    )
    return [f"intervention_representation_not_admitted:{item}" for item in excluded]


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
    design: Mapping[str, Any],
    base_transport: Transport,
) -> tuple[dict[str, Any], ClaudeCallResult | None]:
    call_id = confirmatory._call_id(task_id, arm, role)
    existing = confirmatory._outcome_event(journal, call_id)
    if existing is not None:
        return existing, confirmatory._result_from_dict(existing.get("result"))
    previous = journal.events_for(call_id)
    if any(item["event_kind"] == "message_dispatch" for item in previous):
        outcome = confirmatory._append_outcome(
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
    if confirmatory._arm_usage(journal, arm) >= config["claude"][
        "maximum_total_tokens_per_arm"
    ]:
        outcome = confirmatory._append_outcome(
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
        outcome = confirmatory._append_outcome(
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
    errors = confirmatory._contract_errors(result, role, config, summaries)
    errors.extend(_intervention_contract_errors(result, role, arm, design))
    outcome = confirmatory._append_outcome(
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


def _empty_critic(task_id: str) -> ClaudeCallResult:
    output = ClaudeStructuredOutput(ClaudeRole.CRITIC, task_id, (), ())
    return ClaudeCallResult(
        ClaudeCallStatus.COMPLETED,
        ClaudeRole.CRITIC,
        task_id,
        output,
        {"synthetic_for_branch_construction": True},
    )


def _zero_branch(task_id: str, reason: str, outcomes: Sequence[str]) -> dict[str, Any]:
    body = {
        "behavior_sha256": canonical_sha256({"reason": reason, "task": task_id}),
        "branch_kind": "all_proposals_excluded_outcome",
        "expression": "No hypothesis survived the preregistered arm intervention.",
        "family": "counted_zero_idea_outcome",
        "falsifiers": ["a hypothesis satisfying the arm intervention was retained"],
        "generation_contract_status": "failed",
        "initial_check_status": "failed",
        "invariants": ["no replacement provider call was made"],
        "known_analogues": [],
        "later_used_as_parent": False,
        "llm_origin_assessment": "uncertain",
        "proof_mechanism": "none",
        "proof_mechanism_sha256": canonical_sha256("none"),
        "proof_plan": ["score this arm/task outcome as zero typed usable ideas"],
        "rationale": reason,
        "representation": "other_typed_relation",
        "source_domains": [],
        "synthesis_note": "This placeholder preserves a real zero without repairing the run.",
        "scheduled_outcomes": list(outcomes),
    }
    body["branch_id"] = "branch." + canonical_sha256(body)[:24]
    return body


def _branches(
    task_id: str,
    arm: str,
    proposer: ClaudeCallResult | None,
    critic: ClaudeCallResult | None,
    proposer_outcome: Mapping[str, Any],
    critic_outcome: Mapping[str, Any],
    design: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    outcomes = [proposer_outcome["status"], critic_outcome["status"]]
    if proposer is None or proposer.output is None or not proposer.output.hypotheses:
        return [_zero_branch(task_id, "scheduled proposer produced no admitted hypothesis", outcomes)], 0
    critic = critic or _empty_critic(task_id)
    if critic.output is None:
        return [_zero_branch(task_id, "scheduled critic produced no admitted output", outcomes)], 0
    semantics = design["arms"][arm]["execution_semantics"]
    admitted = set(semantics["admitted_representations"])
    summaries = pilot._candidate_summaries(proposer, arm)
    actions = {item.candidate_id: item.to_dict() for item in critic.output.steering_actions}
    rows: list[dict[str, Any]] = []
    retained_hypotheses = 0
    for hypothesis, summary in zip(proposer.output.hypotheses, summaries, strict=True):
        action = actions.get(summary["candidate_id"])
        if hypothesis.representation not in admitted:
            continue
        if (
            semantics["critic_reject_action"] == "drop_before_expansion"
            and action is not None
            and action["verdict"] == "reject"
        ):
            continue
        retained_hypotheses += 1
        behavior_sha = summary["behavior_sha256"]
        hide_lineage = semantics["origin_lineage_mode"] == "normalize_uncertain_and_hide"
        base = {
            "behavior_sha256": behavior_sha,
            "expression": hypothesis.expression,
            "family": hypothesis.family,
            "falsifiers": list(hypothesis.falsifiers),
            "generation_contract_status": (
                "pass" if outcomes == ["contract_pass", "contract_pass"] else "failed"
            ),
            "initial_check_status": pilot._action_status(action),
            "invariants": list(hypothesis.invariants),
            "known_analogues": [] if hide_lineage else list(hypothesis.known_analogues),
            "later_used_as_parent": False,
            "llm_origin_assessment": (
                "uncertain" if hide_lineage else hypothesis.llm_origin_assessment
            ),
            "rationale": hypothesis.rationale,
            "representation": hypothesis.representation,
            "source_domains": [] if hide_lineage else list(hypothesis.source_idea_domains),
            "synthesis_note": (
                "Origin lineage was hidden by the preregistered intervention."
                if hide_lineage
                else hypothesis.synthesis_note
            ),
            "scheduled_outcomes": outcomes,
        }
        plans = [("llm_declared_plan", list(hypothesis.proof_plan))]
        count = semantics["independent_proof_plans_per_hypothesis"]
        ordered = sorted(
            pilot._PLAN_MECHANISMS,
            key=lambda mechanism: canonical_sha256(
                {"behavior": behavior_sha, "mechanism": mechanism, "task": task_id}
            ),
        )
        plans.extend(
            (
                mechanism,
                [
                    f"apply {mechanism.replace('_', ' ')} independently of the proposed formula",
                    "state premises, boundary cases, and a falsifying obligation",
                    "retain the route until its applicability is tested",
                ],
            )
            for mechanism in ordered[:count]
        )
        for mechanism, plan in plans:
            branch = {
                **base,
                "branch_kind": "hypothesis_proof_route",
                "proof_mechanism": mechanism,
                "proof_mechanism_sha256": canonical_sha256(
                    {"mechanism": mechanism, "steps": plan}
                ),
                "proof_plan": plan,
            }
            branch["branch_id"] = "branch." + canonical_sha256(branch)[:24]
            rows.append(branch)
    if not rows:
        return [
            _zero_branch(
                task_id,
                "all hypotheses were excluded by the preregistered representation or pruning intervention",
                outcomes,
            )
        ], 0
    maximum = semantics["post_generation_recombinations_per_task"]
    if maximum:
        rows.extend(pilot._recombination_branches(rows, task_id, maximum))
        for branch in rows:
            if hide_lineage:
                branch["known_analogues"] = []
                branch["llm_origin_assessment"] = "uncertain"
                branch["source_domains"] = []
                branch[
                    "synthesis_note"
                ] = "Origin lineage was hidden by the preregistered intervention."
            branch.setdefault(
                "generation_contract_status",
                "pass" if outcomes == ["contract_pass", "contract_pass"] else "failed",
            )
            branch.setdefault("scheduled_outcomes", outcomes)
            branch["branch_id"] = "branch." + canonical_sha256(
                {key: value for key, value in branch.items() if key != "branch_id"}
            )[:24]
    if len(rows) > runtime["execution"]["maximum_hypotheses_per_task"] * 3 + maximum:
        raise ComponentKnockoutGenerationError("component-knockout branch budget changed")
    return rows, retained_hypotheses


def _recursive_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden in value or any(_recursive_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_recursive_key(item, forbidden) for item in value)
    return False


def run_generation(
    root: Path,
    *,
    authorization: Mapping[str, Any],
    journal: DurableAttemptJournal,
    credential_file: Path | None = None,
    transport: Transport = urllib_transport,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    experiment = validate_authorization(authorization, root)
    if authorization["journal_path_sha256"] != _journal_path_sha256(journal.path):
        raise ComponentKnockoutGenerationError(
            "authorization does not bind this durable attempt journal path"
        )
    design, dependencies = load_config(root)
    runtime = _runtime_config(design, dependencies, experiment)
    source_bindings = _source_bindings(root)
    if journal.header.get("experiment_id") != experiment["experiment_id"]:
        raise ComponentKnockoutGenerationError("attempt journal experiment binding changed")
    if journal.header.get("source_bindings") != source_bindings:
        raise ComponentKnockoutGenerationError("attempt journal source binding changed")
    arms = (experiment["reference_arm"], experiment["knockout_arm"])
    clients = {
        arm: ClaudeCreativityClient(confirmatory._client_config(runtime), transport) for arm in arms
    }
    confirmatory._restore_client_budgets(clients, journal)
    environment = None
    if credential_file is not None:
        environment = dict(os.environ)
        environment[runtime["claude"]["credential_env_var"]] = ""
        environment["INVARIANT_ENV_FILE"] = str(credential_file.resolve())
    task_results: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        with activated_credential(
            project_root=root,
            env_var=runtime["claude"]["credential_env_var"],
            environment=environment,
        ) as activation:
            for task in sorted(dependencies["generation"]["tasks"], key=lambda item: item["task_id"]):
                task_id = task["task_id"]
                public_task = pilot._public_task(task)
                task_results[task_id] = {}
                proposer_results: dict[str, ClaudeCallResult | None] = {}
                proposer_outcomes: dict[str, dict[str, Any]] = {}
                summaries: dict[str, list[dict[str, Any]]] = {}
                for arm in _ordered_arms(experiment, task_id, ClaudeRole.PROPOSER.value):
                    outcome, result = _run_scheduled(
                        journal,
                        clients[arm],
                        arm=arm,
                        task_id=task_id,
                        role=ClaudeRole.PROPOSER,
                        public_task=public_task,
                        summaries=(),
                        config=runtime,
                        design=design,
                        base_transport=transport,
                    )
                    proposer_outcomes[arm] = outcome
                    proposer_results[arm] = result
                    summaries[arm] = (
                        pilot._candidate_summaries(result, arm)
                        if result is not None and result.output is not None and result.output.hypotheses
                        else confirmatory._sentinel_summary(task_id, arm)
                    )
                for arm in _ordered_arms(experiment, task_id, ClaudeRole.CRITIC.value):
                    critic_outcome, critic = _run_scheduled(
                        journal,
                        clients[arm],
                        arm=arm,
                        task_id=task_id,
                        role=ClaudeRole.CRITIC,
                        public_task=public_task,
                        summaries=summaries[arm],
                        config=runtime,
                        design=design,
                        base_transport=transport,
                    )
                    branches, retained = _branches(
                        task_id,
                        arm,
                        proposer_results[arm],
                        critic,
                        proposer_outcomes[arm],
                        critic_outcome,
                        design,
                        runtime,
                    )
                    tokens = sum(
                        sum(
                            confirmatory._response_usage(
                                journal, confirmatory._call_id(task_id, arm, role)
                            )
                        )
                        for role in (ClaudeRole.PROPOSER, ClaudeRole.CRITIC)
                    )
                    task_results[task_id][arm] = {
                        "branches": branches,
                        "retained_hypotheses": retained,
                        "tokens_used": tokens,
                    }
    except (CredentialActivationError, AttemptJournalError) as error:
        raise ComponentKnockoutGenerationError(str(error)) from error
    call_outcomes = [
        item["payload"] for item in journal.events if item["event_kind"] == "scheduled_call_outcome"
    ]
    dispatches = [
        item["payload"] for item in journal.events if item["event_kind"] == "message_dispatch"
    ]
    if len(call_outcomes) != CALLS_PER_EXPERIMENT:
        raise ComponentKnockoutGenerationError("component-knockout scheduled outcome coverage is incomplete")
    outcomes_by_arm = {
        arm: [item for item in call_outcomes if item["arm"] == arm] for arm in arms
    }
    dispatches_by_arm = {
        arm: [item for item in dispatches if item["arm"] == arm] for arm in arms
    }
    if any(len(outcomes_by_arm[arm]) != 48 for arm in arms):
        raise ComponentKnockoutGenerationError("component-knockout scheduled arm balance changed")
    eligible = (
        all(item["status"] == "contract_pass" for item in call_outcomes)
        and all(len(dispatches_by_arm[arm]) == 48 for arm in arms)
        and all(
            confirmatory._arm_usage(journal, arm)
            <= runtime["claude"]["maximum_total_tokens_per_arm"]
            for arm in arms
        )
    )
    key = journal.unblinding_key
    outputs = []
    mapping = []
    resources = design["matched_resource_budget"]
    for task_id in sorted(task_results):
        paired = []
        for arm in arms:
            blinded_id = pilot._blinded_id(key, task_id, arm)
            result = task_results[task_id][arm]
            paired.append(
                {
                    "blinded_output_id": blinded_id,
                    "branches": sorted(result["branches"], key=lambda item: item["branch_id"]),
                    "resource_budget": {
                        "calls": resources["calls_per_task"],
                        "grammar_depth": resources["grammar_depth"],
                        "tokens": resources["tokens_per_arm"],
                        "verifier_invocations": resources["verifier_invocations_per_task"],
                        "wall_clock_milliseconds": resources["wall_clock_milliseconds_per_call"]
                        * resources["calls_per_task"],
                    },
                    "task_id": task_id,
                    "tokens_used": result["tokens_used"],
                    "typed_usable_ideas": result["retained_hypotheses"],
                }
            )
            mapping.append({"arm": arm, "blinded_output_id": blinded_id, "task_id": task_id})
        outputs.extend(sorted(paired, key=lambda item: item["blinded_output_id"]))
    review = _reseal(
        {
            "schema_version": REVIEW_SCHEMA,
            "suite_id": SUITE_ID,
            "experiment_id": experiment["experiment_id"],
            "removed_feature": experiment["removed_feature"],
            "review_policy": design["review_policy"],
            "review_instructions": (
                "Score every branch independently without inferring which output has the removed "
                "feature. Zero-idea placeholders are real outcomes. Literature novelty is outside "
                "this review."
            ),
            "blinded_outputs": outputs,
            "claims": {
                "arm_identity_disclosed": False,
                "literature_novelty_established": False,
                "more_creative_established": False,
            },
        }
    )
    status_counts: dict[str, int] = {}
    for item in call_outcomes:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    public = _reseal(
        {
            "schema_version": PUBLIC_RECEIPT_SCHEMA,
            "suite_id": SUITE_ID,
            "experiment_id": experiment["experiment_id"],
            "source_bindings": source_bindings,
            "authorization": {
                "authorization_content_sha256": authorization["content_sha256"],
                "authorization_nonce_sha256": hashlib.sha256(
                    bytes.fromhex(str(authorization["authorization_nonce"]))
                ).hexdigest(),
                "authorized_at_utc": authorization["authorized_at_utc"],
                "authorized_by": authorization["authorized_by"],
                "authorized_executions": 1,
                "journal_path_sha256": authorization["journal_path_sha256"],
                "maximum_provider_calls": CALLS_PER_EXPERIMENT,
                "maximum_total_tokens": TOKENS_PER_EXPERIMENT,
                "paid_execution_authorized": True,
            },
            "credential_activation": activation.to_evidence(),
            "intervention": {
                "knockout_arm": experiment["knockout_arm"],
                "reference_arm": experiment["reference_arm"],
                "removed_feature": experiment["removed_feature"],
                "one_feature_removed": True,
                "reference_semantics_sha256": canonical_sha256(
                    design["arms"][experiment["reference_arm"]]["execution_semantics"]
                ),
                "knockout_semantics_sha256": canonical_sha256(
                    design["arms"][experiment["knockout_arm"]]["execution_semantics"]
                ),
            },
            "attempt_accounting": {
                "attempt_journal_content_sha256": journal.content_sha256,
                "balanced_provider_message_attempts": len(dispatches_by_arm[arms[0]])
                == len(dispatches_by_arm[arms[1]]),
                "contract_outcome_counts": dict(sorted(status_counts.items())),
                "provider_message_attempts": len(dispatches),
                "replacement_calls": 0,
                "scheduled_slots": len(call_outcomes),
                "transient_retries": 0,
            },
            "claude_runtime": {
                "authenticated_messages_api_working": bool(dispatches),
                "effort": runtime["claude"]["effort"],
                "model": runtime["claude"]["model"],
                "status": (
                    "PASS_COMPONENT_KNOCKOUT_GENERATION"
                    if eligible
                    else "COUNTED_FAILURES_NOT_CONFIRMATORY"
                ),
                "total_tokens": sum(confirmatory._arm_usage(journal, arm) for arm in arms),
            },
            "blinding": {
                "review_packet_content_sha256": review["content_sha256"],
                "unblinding_key_sha256": hashlib.sha256(key).hexdigest(),
                "status": "SEALED_UNTIL_TWO_NAMED_REVIEWS",
            },
            "release_gate": {
                "generation_eligible": eligible,
                "named_blinded_reviews_complete": False,
                "status": "BLOCKED_TWO_NAMED_BLINDED_REVIEWS_REQUIRED",
                "tournament_scored": False,
            },
            "claims": {
                "credential_material_persisted": False,
                "literature_novelty_established": False,
                "more_creative_established": False,
            },
        }
    )
    coordinator = _reseal(
        {
            "schema_version": COORDINATOR_SCHEMA,
            "suite_id": SUITE_ID,
            "experiment_id": experiment["experiment_id"],
            "authorization_content_sha256": authorization["content_sha256"],
            "public_receipt_content_sha256": public["content_sha256"],
            "review_packet_content_sha256": review["content_sha256"],
            "attempt_journal_content_sha256": journal.content_sha256,
            "unblinding_key_hex": key.hex(),
            "mapping": sorted(
                mapping, key=lambda item: (item["task_id"], item["blinded_output_id"])
            ),
            "arm_outcome_status_counts": {
                arm: {
                    status: sum(item["status"] == status for item in outcomes_by_arm[arm])
                    for status in sorted({row["status"] for row in call_outcomes})
                }
                for arm in arms
            },
            "claims": {"safe_to_publish_before_review": False},
        }
    )
    validate_public(review, public, root)
    validate_coordinator(coordinator, review, public, journal)
    return review, public, coordinator


def validate_public(
    review: Mapping[str, Any], public: Mapping[str, Any], root: Path | None = None
) -> None:
    for value, schema, label in (
        (review, REVIEW_SCHEMA, "component-knockout review packet"),
        (public, PUBLIC_RECEIPT_SCHEMA, "component-knockout public receipt"),
    ):
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("schema_version") != schema or value.get("content_sha256") != canonical_sha256(
            body
        ):
            raise ComponentKnockoutGenerationError(f"{label} identity or seal changed")
    if (
        review.get("suite_id") != SUITE_ID
        or public.get("suite_id") != SUITE_ID
        or review.get("experiment_id") != public.get("experiment_id")
        or review.get("removed_feature") != public.get("intervention", {}).get("removed_feature")
        or _recursive_key(review, "arm")
    ):
        raise ComponentKnockoutGenerationError("component-knockout public identity or blinding changed")
    outputs = review.get("blinded_outputs", [])
    if len(outputs) != 48 or len({item["blinded_output_id"] for item in outputs}) != 48:
        raise ComponentKnockoutGenerationError("component-knockout review output coverage changed")
    counts: dict[str, int] = {}
    for item in outputs:
        counts[item["task_id"]] = counts.get(item["task_id"], 0) + 1
        if (
            not isinstance(item.get("tokens_used"), int)
            or item["tokens_used"] < 0
            or not isinstance(item.get("typed_usable_ideas"), int)
            or item["typed_usable_ideas"] < 0
            or not item.get("branches")
        ):
            raise ComponentKnockoutGenerationError("component-knockout review output is invalid")
    if len(counts) != 24 or set(counts.values()) != {2}:
        raise ComponentKnockoutGenerationError("component-knockout paired task coverage changed")
    attempts = public.get("attempt_accounting", {})
    release = public.get("release_gate", {})
    credential = public.get("credential_activation", {})
    authorization = public.get("authorization", {})
    if (
        attempts.get("scheduled_slots") != CALLS_PER_EXPERIMENT
        or attempts.get("provider_message_attempts", CALLS_PER_EXPERIMENT + 1)
        > CALLS_PER_EXPERIMENT
        or attempts.get("replacement_calls") != 0
        or attempts.get("transient_retries") != 0
        or release.get("named_blinded_reviews_complete") is not False
        or release.get("tournament_scored") is not False
        or credential.get("credential_persisted") is not False
        or credential.get("credential_value_recorded") is not False
        or authorization.get("paid_execution_authorized") is not True
        or authorization.get("authorized_executions") != 1
        or authorization.get("maximum_provider_calls") != CALLS_PER_EXPERIMENT
        or authorization.get("maximum_total_tokens") != TOKENS_PER_EXPERIMENT
        or any(review.get("claims", {}).values())
        or any(public.get("claims", {}).values())
    ):
        raise ComponentKnockoutGenerationError("component-knockout accounting or release gate changed")
    if root is not None:
        root = root.resolve()
        design, _ = load_config(root)
        experiment = _experiment(design, str(public["experiment_id"]))
        if public.get("source_bindings") != _source_bindings(root):
            raise ComponentKnockoutGenerationError("component-knockout source bindings changed")
        intervention = public.get("intervention", {})
        if (
            intervention.get("reference_arm") != experiment["reference_arm"]
            or intervention.get("knockout_arm") != experiment["knockout_arm"]
            or intervention.get("removed_feature") != experiment["removed_feature"]
            or intervention.get("one_feature_removed") is not True
        ):
            raise ComponentKnockoutGenerationError("component-knockout intervention binding changed")


def validate_coordinator(
    coordinator: Mapping[str, Any],
    review: Mapping[str, Any],
    public: Mapping[str, Any],
    journal: DurableAttemptJournal,
) -> None:
    body = {key: item for key, item in coordinator.items() if key != "content_sha256"}
    if (
        coordinator.get("schema_version") != COORDINATOR_SCHEMA
        or coordinator.get("suite_id") != SUITE_ID
        or coordinator.get("content_sha256") != canonical_sha256(body)
        or coordinator.get("public_receipt_content_sha256") != public["content_sha256"]
        or coordinator.get("review_packet_content_sha256") != review["content_sha256"]
        or coordinator.get("attempt_journal_content_sha256") != journal.content_sha256
        or coordinator.get("authorization_content_sha256")
        != public["authorization"]["authorization_content_sha256"]
        or len(coordinator.get("mapping", [])) != 48
        or coordinator.get("claims") != {"safe_to_publish_before_review": False}
    ):
        raise ComponentKnockoutGenerationError("component-knockout private coordinator changed")
    key = bytes.fromhex(str(coordinator["unblinding_key_hex"]))
    if key != journal.unblinding_key or hashlib.sha256(key).hexdigest() != public["blinding"][
        "unblinding_key_sha256"
    ]:
        raise ComponentKnockoutGenerationError("component-knockout unblinding key changed")
    for item in coordinator["mapping"]:
        if pilot._blinded_id(key, item["task_id"], item["arm"]) != item["blinded_output_id"]:
            raise ComponentKnockoutGenerationError("component-knockout unblinding mapping changed")


def _private_path(root: Path, path: Path) -> Path:
    return confirmatory._private_path(root, path)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("authorization-template")
    template.add_argument("--root", type=Path, default=Path.cwd())
    template.add_argument("--experiment-id", required=True)
    template.add_argument("--journal", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    seal = subparsers.add_parser("seal-authorization")
    seal.add_argument("--input", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--authorization", type=Path, required=True)
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
    if args.command == "authorization-template":
        root = args.root.resolve()
        journal_arg = args.journal if args.journal.is_absolute() else root / args.journal
        journal_path = _private_path(root, journal_arg)
        value = authorization_template(root, args.experiment_id, journal_path)
        _write(args.output, value)
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.command == "seal-authorization":
        value = seal_authorization(_read_json(args.input, "authorization draft"))
        _write(args.output, value)
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.command == "validate":
        review = _read_json(args.review_packet, "component-knockout review packet")
        receipt = _read_json(args.receipt, "component-knockout public receipt")
        validate_public(review, receipt, args.root.resolve())
        print(
            json.dumps(
                {
                    "content_sha256": receipt["content_sha256"],
                    "experiment_id": receipt["experiment_id"],
                    "provider_message_attempts": receipt["attempt_accounting"][
                        "provider_message_attempts"
                    ],
                    "status": receipt["release_gate"]["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    root = args.root.resolve()
    authorization = _read_json(args.authorization, "component-knockout authorization")
    experiment = validate_authorization(authorization, root)
    journal_arg = args.journal if args.journal.is_absolute() else root / args.journal
    coordinator_arg = (
        args.coordinator_output
        if args.coordinator_output.is_absolute()
        else root / args.coordinator_output
    )
    journal_path = _private_path(root, journal_arg)
    coordinator_path = _private_path(root, coordinator_arg)
    if journal_path == coordinator_path:
        raise ComponentKnockoutGenerationError("attempt journal and coordinator paths must differ")
    source_bindings = _source_bindings(root)
    if journal_path.exists():
        journal = DurableAttemptJournal.load(journal_path)
    else:
        journal = DurableAttemptJournal.create(
            journal_path,
            experiment_id=experiment["experiment_id"],
            source_bindings=source_bindings,
            unblinding_key=secrets.token_bytes(32),
        )
    review, receipt, coordinator = run_generation(
        root,
        authorization=authorization,
        journal=journal,
        credential_file=args.credential_file,
    )
    _write(args.review_output, review)
    _write(args.receipt_output, receipt)
    _write(coordinator_path, coordinator)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "experiment_id": receipt["experiment_id"],
                "generation_eligible": receipt["release_gate"]["generation_eligible"],
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
