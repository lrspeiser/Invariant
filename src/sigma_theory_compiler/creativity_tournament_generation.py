"""Run paired blind Claude generation and emit an arm-free human review packet.

The runner reads only the committed generation packet, never the holdout packet.  Baseline and
treatment use the same model, effort, task order distribution, calls, output ceiling, total-token
ceiling, timeout, grammar depth, and verifier allocation.  Arm identity and raw call traces remain
in an ignored coordinator packet until two named reviewers finish scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from .claude_creativity_api import (
    ClaudeAPIConfig,
    ClaudeBudget,
    ClaudeCallResult,
    ClaudeCallStatus,
    ClaudeCreativityClient,
    ClaudeCreativityError,
    ClaudeRole,
    Transport,
    urllib_transport,
)
from .core_credential import CredentialActivationError, activated_credential
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/creativity_tournament_generation.json"
RUNNER_PATH = "src/sigma_theory_compiler/creativity_tournament_generation.py"
CLAUDE_ADAPTER_PATH = "src/sigma_theory_compiler/claude_creativity_api.py"
CONFIG_SCHEMA = "invariant-creativity-tournament-generation-config-1.0"
REVIEW_SCHEMA = "invariant-creativity-tournament-review-packet-1.0"
PUBLIC_RECEIPT_SCHEMA = "invariant-creativity-tournament-public-generation-1.0"
COORDINATOR_SCHEMA = "invariant-creativity-tournament-private-coordinator-1.0"
CHECKPOINT_SCHEMA = "invariant-creativity-tournament-private-checkpoint-1.0"
_ARMS = ("baseline", "full_creativity_first")
_PLAN_MECHANISMS = (
    "strengthened_induction",
    "bijection_or_involution",
    "minimal_counterexample_descent",
    "transform_and_extract",
    "contradiction_via_invariant",
    "variational_or_dual_certificate",
)
_HEX = frozenset("0123456789abcdef")
# Complete generation-time bundle sealed by the published 96-call pilot receipt. Historical
# bundles remain evidence of the code that produced a run; they are never rewritten as current code.
_HISTORICAL_SOURCE_SHA256_BUNDLES = (
    {
        "claude_adapter": "a37046bc911f4d4400d120de6f94077a55c3dc3896f4aecc2b1b24f0a9e7ce84",
        "config": "700162b09b4d01b47855b17cf57a980ec1fc03db0a0e473f63d0a6d278d06c39",
        "runner": "7b8d5cf01db3c85d8b6dc147cd5ee0511da646145b2291f21a1f4cab1e6a8460",
    },
)


class TournamentGenerationError(ValueError):
    """The paired generation, blinding, budget, or no-holdout invariant failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TournamentGenerationError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise TournamentGenerationError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _git_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in _HEX for character in value)
    ):
        raise TournamentGenerationError(f"{label} is not a full lowercase Git object ID")
    return value


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _source_bindings(root: Path) -> dict[str, Any]:
    paths = {
        "claude_adapter": CLAUDE_ADAPTER_PATH,
        "config": CONFIG_PATH,
        "runner": RUNNER_PATH,
    }
    return {
        name: {"path": path, "sha256": _normalized_file_sha256(root / path)}
        for name, path in sorted(paths.items())
    }


def _compatible_source_bindings(root: Path) -> tuple[dict[str, Any], ...]:
    """Return the current bundle plus explicitly preserved generation-time bundles."""

    current = _source_bindings(root)
    historical = tuple(
        {
            name: {"path": current[name]["path"], "sha256": sha256}
            for name, sha256 in sorted(bundle.items())
        }
        for bundle in _HISTORICAL_SOURCE_SHA256_BUNDLES
    )
    return (current, *historical)


def load_config(root: Path) -> dict[str, Any]:
    value = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    _strict(
        value,
        {
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
        "tournament config",
    )
    if value["schema_version"] != CONFIG_SCHEMA or value["experiment_id"] != (
        "creativity-first-vs-falsification-first-001"
    ):
        raise TournamentGenerationError("tournament config identity changed")
    for key in ("baseline_commit", "treatment_commit"):
        _git_commit(value[key], key)
    if value["baseline_commit"] == value["treatment_commit"]:
        raise TournamentGenerationError("baseline and treatment commits are not distinct")
    packet = value["generation_packet"]
    _strict(packet, {"content_sha256", "path", "required_tasks"}, "generation packet binding")
    _sha(packet["content_sha256"], "generation packet content hash")
    if packet["required_tasks"] < 24 or "target" in packet["path"].lower():
        raise TournamentGenerationError("generation packet binding is too small or target-bearing")
    claude = value["claude"]
    _strict(
        claude,
        {
            "credential_env_var",
            "effort",
            "maximum_calls_per_arm",
            "maximum_output_tokens_per_call",
            "maximum_total_tokens_per_arm",
            "model",
            "timeout_seconds",
        },
        "tournament Claude policy",
    )
    if (
        claude["credential_env_var"] != "ANTHROPIC_API_KEY"
        or claude["model"] != "claude-opus-4-6"
        or claude["effort"] != "high"
        or claude["maximum_calls_per_arm"] != 2 * packet["required_tasks"]
    ):
        raise TournamentGenerationError("tournament Claude pairing policy changed")
    resource = value["matched_resource_budget"]
    _strict(
        resource,
        {
            "calls_per_task",
            "grammar_depth",
            "tokens_per_arm",
            "verifier_invocations_per_task",
            "wall_clock_milliseconds_per_call",
        },
        "matched resource budget",
    )
    if (
        resource["calls_per_task"] != 2
        or resource["tokens_per_arm"] != claude["maximum_total_tokens_per_arm"]
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in resource.values())
    ):
        raise TournamentGenerationError("tournament resources are not fully matched")
    prompts = value["prompt_policies"]
    if set(prompts) != set(_ARMS):
        raise TournamentGenerationError("tournament prompt arms changed")
    for arm in _ARMS:
        _strict(
            prompts[arm],
            {"critic_instruction", "proposer_instruction", "system_instruction"},
            "tournament prompt policy",
        )
        if any(not isinstance(item, str) or len(item) < 40 for item in prompts[arm].values()):
            raise TournamentGenerationError("tournament prompt policy is empty")
    execution = value["execution"]
    _strict(
        execution,
        {
            "arm_order_seed",
            "maximum_hypotheses_per_task",
            "requested_hypotheses_per_task",
            "treatment_independent_plans_per_hypothesis",
            "treatment_recombinations_per_task",
        },
        "tournament execution policy",
    )
    if (
        not execution["arm_order_seed"].startswith("sha256:")
        or execution["requested_hypotheses_per_task"] != 3
        or execution["maximum_hypotheses_per_task"] > 8
    ):
        raise TournamentGenerationError("tournament execution bounds changed")
    review = value["review"]
    _strict(
        review,
        {"axes", "minimum_named_reviewers", "rating_scale", "useful_threshold_each_axis"},
        "tournament review policy",
    )
    if (
        review["minimum_named_reviewers"] < 2
        or review["axes"] != ["coherence", "nontriviality", "followup_value"]
        or review["useful_threshold_each_axis"] != 3
    ):
        raise TournamentGenerationError("tournament review policy weakened")
    return value


def _load_generation_packet(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["generation_packet"]
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise TournamentGenerationError("generation packet escapes repository") from error
    packet = json.loads(path.read_text(encoding="utf-8"))
    if (
        packet.get("content_sha256") != binding["content_sha256"]
        or packet.get("schema_version") != "invariant-rotating-external-generation-packet-1.0"
        or len(packet.get("tasks", [])) != binding["required_tasks"]
    ):
        raise TournamentGenerationError("generation packet binding changed")
    serialized = json.dumps(packet, sort_keys=True).lower()
    if '"holdout"' in serialized or '"source_uri"' in serialized or '"source_id"' in serialized:
        raise TournamentGenerationError("generation packet contains sealed source or holdout material")
    return packet


def _checkpoint_body(
    root: Path,
    generation: Mapping[str, Any],
    unblinding_key: bytes,
    calls: Mapping[str, list[dict[str, Any]]],
    task_results: Mapping[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config_sha256": _normalized_file_sha256(root / CONFIG_PATH),
        "runner_sha256": _normalized_file_sha256(root / RUNNER_PATH),
        "claude_adapter_sha256": _normalized_file_sha256(root / CLAUDE_ADAPTER_PATH),
        "generation_packet_content_sha256": generation["content_sha256"],
        "unblinding_key_hex": unblinding_key.hex(),
        "completed_task_ids": sorted(task_results),
        "arm_calls": {arm: list(calls[arm]) for arm in _ARMS},
        "task_results": dict(task_results),
        "claims": {"safe_to_publish": False, "target_reads": 0},
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def _write_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    root: Path,
    generation: Mapping[str, Any],
    unblinding_key: bytes,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("schema_version") != CHECKPOINT_SCHEMA
        or value.get("content_sha256") != canonical_sha256(body)
        or value.get("config_sha256") != _normalized_file_sha256(root / CONFIG_PATH)
        or value.get("runner_sha256") != _normalized_file_sha256(root / RUNNER_PATH)
        or value.get("claude_adapter_sha256")
        != _normalized_file_sha256(root / CLAUDE_ADAPTER_PATH)
        or value.get("generation_packet_content_sha256") != generation["content_sha256"]
        or value.get("unblinding_key_hex") != unblinding_key.hex()
        or value.get("claims") != {"safe_to_publish": False, "target_reads": 0}
    ):
        raise TournamentGenerationError("private tournament checkpoint binding changed")
    calls = value.get("arm_calls")
    task_results = value.get("task_results")
    completed = value.get("completed_task_ids")
    generation_ids = {item["task_id"] for item in generation["tasks"]}
    if (
        not isinstance(calls, Mapping)
        or set(calls) != set(_ARMS)
        or not isinstance(task_results, Mapping)
        or sorted(task_results) != completed
        or not set(task_results).issubset(generation_ids)
    ):
        raise TournamentGenerationError("private tournament checkpoint coverage changed")
    for arm in _ARMS:
        arm_calls = calls[arm]
        if (
            not isinstance(arm_calls, list)
            or len(arm_calls) != 2 * len(completed)
            or any(item.get("status") != "completed" for item in arm_calls)
        ):
            raise TournamentGenerationError("private tournament checkpoint call count changed")
        role_counts = {
            role: sum(item.get("role") == role for item in arm_calls)
            for role in ("proposer", "critic")
        }
        if set(role_counts.values()) != {len(completed)}:
            raise TournamentGenerationError("private tournament checkpoint role pairing changed")
    if any(set(task_results[task_id]) != set(_ARMS) for task_id in completed):
        raise TournamentGenerationError("private tournament checkpoint arm pairing changed")
    return ({arm: list(calls[arm]) for arm in _ARMS}, dict(task_results))


def _restore_budget(client: ClaudeCreativityClient, calls: Sequence[Mapping[str, Any]]) -> None:
    input_tokens = sum(item["evidence"]["usage"]["input_tokens"] for item in calls)
    output_tokens = sum(item["evidence"]["usage"]["output_tokens"] for item in calls)
    client.budget = ClaudeBudget(len(calls), input_tokens, output_tokens)
    if client.budget.total_tokens > client.config.maximum_total_tokens:
        raise TournamentGenerationError("private checkpoint exceeds the Claude token budget")


def _client_config(config: Mapping[str, Any]) -> ClaudeAPIConfig:
    claude = config["claude"]
    return ClaudeAPIConfig(
        model=claude["model"],
        credential_env_var=claude["credential_env_var"],
        execution_enabled=True,
        maximum_calls=claude["maximum_calls_per_arm"],
        maximum_total_tokens=claude["maximum_total_tokens_per_arm"],
        maximum_output_tokens=claude["maximum_output_tokens_per_call"],
        timeout_seconds=claude["timeout_seconds"],
        effort=claude["effort"],
    )


def _retrying_transport(transport: Transport) -> Transport:
    """Retry only transient network/service failures, never invalid mathematical output."""

    def wrapped(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        delays = (1, 2, 4, 8)
        for attempt in range(len(delays) + 1):
            try:
                return transport(method, url, headers, body, timeout)
            except ClaudeCreativityError as error:
                message = str(error)
                retryable = "transport failed" in message or any(
                    f"HTTP {status}" in message for status in (429, 500, 502, 503, 529)
                )
                if not retryable or attempt == len(delays):
                    raise
                time.sleep(delays[attempt])
        raise AssertionError("unreachable retry loop")

    return wrapped


def _public_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "representation_family": task["representation_family"],
        "training": [dict(item) for item in task["training"]],
        "request": (
            "Propose structural explanations, formulas or constructions, and proof routes for "
            "the visible indexed values. Do not assume a unique continuation."
        ),
    }


def _arm_order(config: Mapping[str, Any], task_id: str, phase: str) -> tuple[str, ...]:
    seed = config["execution"]["arm_order_seed"]
    return tuple(
        sorted(
            _ARMS,
            key=lambda arm: hashlib.sha256(f"{seed}:{task_id}:{phase}:{arm}".encode()).digest(),
        )
    )


def _candidate_summaries(call: ClaudeCallResult, arm: str) -> list[dict[str, Any]]:
    if call.output is None:
        raise TournamentGenerationError("completed proposer call has no structured output")
    rows = []
    for hypothesis in call.output.hypotheses:
        body = hypothesis.to_dict()
        rows.append(
            {
                "behavior_sha256": canonical_sha256(
                    {
                        "expression": hypothesis.expression,
                        "invariants": list(hypothesis.invariants),
                        "representation": hypothesis.representation,
                    }
                ),
                "candidate_id": "candidate."
                + canonical_sha256({"arm": arm, "hypothesis": body})[:24],
                "expression": hypothesis.expression,
                "representation": hypothesis.representation,
                "typed_status": "SCHEMA_ADMITTED_EXECUTION_UNTESTED",
            }
        )
    return rows


def _validate_proposer(call: ClaudeCallResult, config: Mapping[str, Any]) -> None:
    if call.output is None:
        raise TournamentGenerationError("completed proposer call has no structured output")
    expected = config["execution"]["requested_hypotheses_per_task"]
    if (
        len(call.output.hypotheses) != expected
        or call.output.steering_actions
        or call.output.rejected_hypotheses
        or call.output.rejected_steering_actions
    ):
        raise TournamentGenerationError(
            "tournament proposer did not return the exact admitted hypothesis allocation: "
            f"expected={expected}, admitted={len(call.output.hypotheses)}, "
            f"quarantined={call.output.rejected_hypotheses}, "
            f"inapplicable_actions={len(call.output.steering_actions)}"
        )


def _validate_critic(
    call: ClaudeCallResult, summaries: Sequence[Mapping[str, Any]]
) -> None:
    if call.output is None:
        raise TournamentGenerationError("completed critic call has no structured output")
    expected = [item["candidate_id"] for item in summaries]
    actual = [item.candidate_id for item in call.output.steering_actions]
    if (
        call.output.hypotheses
        or call.output.rejected_hypotheses
        or call.output.rejected_steering_actions
        or len(actual) != len(expected)
        or len(set(actual)) != len(actual)
        or set(actual) != set(expected)
    ):
        raise TournamentGenerationError(
            "tournament critic did not cover every admitted candidate exactly once: "
            f"expected={len(expected)}, admitted={len(actual)}, "
            f"quarantined={call.output.rejected_steering_actions}, "
            f"critic_hypotheses={len(call.output.hypotheses)}, "
            f"unique_ids={len(set(actual))}, ids_match={set(actual) == set(expected)}"
        )


def _action_status(action: Mapping[str, Any] | None) -> str:
    if action is None or action.get("verdict") == "retain":
        return "untested"
    if action.get("verdict") == "repair":
        return "blocked"
    return "failed"


def _hypothesis_branches(
    proposer: ClaudeCallResult,
    critic: ClaudeCallResult,
    arm: str,
    task_id: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if proposer.output is None or critic.output is None:
        raise TournamentGenerationError("completed tournament call lacks structured output")
    summaries = _candidate_summaries(proposer, arm)
    if len(proposer.output.hypotheses) > config["execution"]["maximum_hypotheses_per_task"]:
        raise TournamentGenerationError("tournament proposer exceeded hypothesis bound")
    actions = {item.candidate_id: item.to_dict() for item in critic.output.steering_actions}
    rows = []
    for hypothesis, summary in zip(proposer.output.hypotheses, summaries, strict=True):
        action = actions.get(summary["candidate_id"])
        if arm == "baseline" and action is not None and action["verdict"] == "reject":
            continue
        behavior_sha = summary["behavior_sha256"]
        base = {
            "behavior_sha256": behavior_sha,
            "expression": hypothesis.expression,
            "family": hypothesis.family,
            "falsifiers": list(hypothesis.falsifiers),
            "initial_check_status": _action_status(action),
            "invariants": list(hypothesis.invariants),
            "known_analogues": list(hypothesis.known_analogues),
            "later_used_as_parent": False,
            "llm_origin_assessment": hypothesis.llm_origin_assessment,
            "rationale": hypothesis.rationale,
            "representation": hypothesis.representation,
            "source_domains": list(hypothesis.source_idea_domains),
            "synthesis_note": hypothesis.synthesis_note,
        }
        plans = [("llm_declared_plan", list(hypothesis.proof_plan))]
        if arm == "full_creativity_first":
            count = config["execution"]["treatment_independent_plans_per_hypothesis"]
            ordered = sorted(
                _PLAN_MECHANISMS,
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
            proof_sha = canonical_sha256({"mechanism": mechanism, "steps": plan})
            branch = {
                **base,
                "branch_kind": "hypothesis_proof_route",
                "proof_mechanism": mechanism,
                "proof_mechanism_sha256": proof_sha,
                "proof_plan": plan,
            }
            branch["branch_id"] = "branch." + canonical_sha256(branch)[:24]
            rows.append(branch)
    return rows


def _recombination_branches(
    rows: list[dict[str, Any]], task_id: str, maximum: int
) -> list[dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    for row in rows:
        parents.setdefault(row["behavior_sha256"], row)
    recombinations = []
    for left, right in combinations(parents.values(), 2):
        domains = sorted(set(left["source_domains"]) | set(right["source_domains"]))
        expression = (
            f"Transport invariants from ({left['expression']}) into the representation of "
            f"({right['expression']}), then test both directions."
        )[:1000]
        behavior_sha = canonical_sha256(
            {
                "left": left["behavior_sha256"],
                "right": right["behavior_sha256"],
                "task": task_id,
            }
        )
        proof_plan = [
            "identify a shared invariant or conserved structure",
            "translate it across the two representations",
            "test both parent specializations and a separating counterexample",
        ]
        branch = {
            "behavior_sha256": behavior_sha,
            "branch_kind": "cross_idea_recombination",
            "expression": expression,
            "family": "cross_domain_recombination",
            "falsifiers": ["either parent specialization fails", "translation loses the invariant"],
            "initial_check_status": "untested",
            "invariants": sorted(set(left["invariants"]) | set(right["invariants"]))[:12],
            "known_analogues": sorted(set(left["known_analogues"]) | set(right["known_analogues"]))[:12],
            "later_used_as_parent": False,
            "llm_origin_assessment": "cross_domain_synthesis",
            "proof_mechanism": "representation_transport_and_bidirectional_test",
            "proof_mechanism_sha256": canonical_sha256(proof_plan),
            "proof_plan": proof_plan,
            "rationale": "Recombine retained parent ideas before verifier status can prune either lineage.",
            "representation": "transform_relation",
            "source_domains": domains,
            "synthesis_note": "Post-generation recombination; literature novelty is not implied.",
        }
        branch["branch_id"] = "branch." + canonical_sha256(branch)[:24]
        recombinations.append(branch)
        left["later_used_as_parent"] = True
        right["later_used_as_parent"] = True
        if len(recombinations) >= maximum:
            break
    return recombinations


def _blinded_id(key: bytes, task_id: str, arm: str) -> str:
    return "output." + hmac.new(key, f"{task_id}:{arm}".encode(), hashlib.sha256).hexdigest()[:24]


def run_generation(
    root: Path,
    *,
    unblinding_key: bytes,
    credential_file: Path | None = None,
    transport: Transport = urllib_transport,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    if len(unblinding_key) < 32:
        raise TournamentGenerationError("unblinding key is shorter than 256 bits")
    config = load_config(root)
    generation = _load_generation_packet(root, config)
    retrying_transport = _retrying_transport(transport)
    clients = {
        arm: ClaudeCreativityClient(_client_config(config), retrying_transport) for arm in _ARMS
    }
    calls: dict[str, list[dict[str, Any]]] = {arm: [] for arm in _ARMS}
    task_results: dict[str, dict[str, dict[str, Any]]] = {}
    if checkpoint_path is not None and checkpoint_path.is_file():
        calls, task_results = _load_checkpoint(
            checkpoint_path, root, generation, unblinding_key
        )
        for arm in _ARMS:
            _restore_budget(clients[arm], calls[arm])
    environment = None
    if credential_file is not None:
        environment = dict(os.environ)
        environment[config["claude"]["credential_env_var"]] = ""
        environment["INVARIANT_ENV_FILE"] = str(credential_file.resolve())
    try:
        with activated_credential(
            project_root=root,
            env_var=config["claude"]["credential_env_var"],
            environment=environment,
        ) as activation:
            for task in generation["tasks"]:
                task_id = task["task_id"]
                if task_id in task_results:
                    continue
                public = _public_task(task)
                task_results[task_id] = {}
                proposer_calls: dict[str, ClaudeCallResult] = {}
                summaries: dict[str, list[dict[str, Any]]] = {}
                for arm in _arm_order(config, task_id, "proposer"):
                    policy = config["prompt_policies"][arm]
                    call = clients[arm].run(
                        ClaudeRole.PROPOSER,
                        task_id,
                        public,
                        instruction_override=policy["proposer_instruction"],
                        system_override=policy["system_instruction"],
                        hypothesis_slots=config["execution"]["requested_hypotheses_per_task"],
                    )
                    if call.status is not ClaudeCallStatus.COMPLETED:
                        raise TournamentGenerationError("Claude proposer did not complete")
                    _validate_proposer(call, config)
                    proposer_calls[arm] = call
                    summaries[arm] = _candidate_summaries(call, arm)
                    calls[arm].append(call.to_dict())
                for arm in _arm_order(config, task_id, "critic"):
                    policy = config["prompt_policies"][arm]
                    if not summaries[arm]:
                        raise TournamentGenerationError("Claude proposer returned no hypotheses")
                    critic = clients[arm].run(
                        ClaudeRole.CRITIC,
                        task_id,
                        public,
                        candidate_summaries=summaries[arm],
                        instruction_override=policy["critic_instruction"],
                        system_override=policy["system_instruction"],
                    )
                    if critic.status is not ClaudeCallStatus.COMPLETED:
                        raise TournamentGenerationError("Claude critic did not complete")
                    _validate_critic(critic, summaries[arm])
                    calls[arm].append(critic.to_dict())
                    branches = _hypothesis_branches(
                        proposer_calls[arm], critic, arm, task_id, config
                    )
                    if arm == "full_creativity_first":
                        branches.extend(
                            _recombination_branches(
                                branches,
                                task_id,
                                config["execution"]["treatment_recombinations_per_task"],
                            )
                        )
                    task_results[task_id][arm] = {
                        "branches": branches,
                        "tokens_used": sum(
                            item["evidence"]["usage"]["input_tokens"]
                            + item["evidence"]["usage"]["output_tokens"]
                            for item in calls[arm][-2:]
                        ),
                    }
                if checkpoint_path is not None:
                    _write_checkpoint(
                        checkpoint_path,
                        _checkpoint_body(root, generation, unblinding_key, calls, task_results),
                    )
    except CredentialActivationError as error:
        raise TournamentGenerationError(str(error)) from error
    for arm in _ARMS:
        if clients[arm].budget.calls != config["claude"]["maximum_calls_per_arm"]:
            raise TournamentGenerationError("paired Claude call counts are not complete")
    resource = config["matched_resource_budget"]
    mapping = []
    blinded_outputs = []
    for task_id in sorted(task_results):
        per_task = []
        for arm in _ARMS:
            blinded_id = _blinded_id(unblinding_key, task_id, arm)
            result = task_results[task_id][arm]
            branches = sorted(result["branches"], key=lambda item: item["branch_id"])
            per_task.append(
                {
                    "blinded_output_id": blinded_id,
                    "branches": branches,
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
                    "typed_usable_ideas": len(branches),
                }
            )
            mapping.append({"arm": arm, "blinded_output_id": blinded_id, "task_id": task_id})
        blinded_outputs.extend(sorted(per_task, key=lambda item: item["blinded_output_id"]))
    review_body: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA,
        "experiment_id": config["experiment_id"],
        "review_policy": config["review"],
        "review_instructions": (
            "Score every branch independently without inferring system identity. A useful branch "
            "must score at least 3 on coherence, nontriviality, and follow-up value from each "
            "named reviewer. Known rewrites may still be useful; do not score literature novelty."
        ),
        "blinded_outputs": blinded_outputs,
        "claims": {
            "arm_identity_disclosed": False,
            "holdout_opened": False,
            "literature_novelty_established": False,
            "more_creative_established": False,
        },
    }
    review_body["content_sha256"] = canonical_sha256(review_body)
    all_calls = sorted(
        [item for arm in _ARMS for item in calls[arm]],
        key=lambda item: (item["benchmark_id"], item["role"], item["evidence"]["api_response_id"]),
    )
    public_body: dict[str, Any] = {
        "schema_version": PUBLIC_RECEIPT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "source_bindings": {
            **_source_bindings(root),
            "generation_packet_content_sha256": generation["content_sha256"],
        },
        "credential_activation": activation.to_evidence(),
        "claude_runtime": {
            "authenticated_messages_api_working": True,
            "call_evidence_root_sha256": canonical_sha256(all_calls),
            "completed_calls": sum(len(items) for items in calls.values()),
            "effort": config["claude"]["effort"],
            "model": config["claude"]["model"],
            "status": "PASS_PAIRED_GENERATION",
            "total_tokens": sum(client.budget.total_tokens for client in clients.values()),
        },
        "blinding": {
            "review_packet_content_sha256": review_body["content_sha256"],
            "unblinding_key_sha256": hashlib.sha256(unblinding_key).hexdigest(),
            "status": "SEALED_UNTIL_TWO_NAMED_REVIEWS",
        },
        "chronology": [
            {"event": "generation_packet_loaded", "sequence": 0, "target_reads": 0},
            {"event": "paired_proposals_and_critiques_completed", "sequence": 1, "target_reads": 0},
            {"event": "review_packet_and_unblinding_commitment_sealed", "sequence": 2, "target_reads": 0},
        ],
        "release_gate": {
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
    public_body["content_sha256"] = canonical_sha256(public_body)
    coordinator_body: dict[str, Any] = {
        "schema_version": COORDINATOR_SCHEMA,
        "experiment_id": config["experiment_id"],
        "public_receipt_content_sha256": public_body["content_sha256"],
        "review_packet_content_sha256": review_body["content_sha256"],
        "unblinding_key_hex": unblinding_key.hex(),
        "mapping": sorted(mapping, key=lambda item: (item["task_id"], item["blinded_output_id"])),
        "arm_calls": calls,
        "call_evidence_root_sha256": canonical_sha256(all_calls),
        "claims": {"safe_to_publish_before_review": False},
    }
    coordinator_body["content_sha256"] = canonical_sha256(coordinator_body)
    validate_public_generation(review_body, public_body, root)
    validate_coordinator(coordinator_body, review_body, public_body)
    return review_body, public_body, coordinator_body


def validate_public_generation(
    review: Mapping[str, Any], receipt: Mapping[str, Any], root: Path | None = None
) -> None:
    for value, schema, label in (
        (review, REVIEW_SCHEMA, "tournament review packet"),
        (receipt, PUBLIC_RECEIPT_SCHEMA, "tournament public receipt"),
    ):
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("schema_version") != schema or value.get("content_sha256") != canonical_sha256(body):
            raise TournamentGenerationError(f"{label} identity or content seal changed")
    outputs = review.get("blinded_outputs", [])
    if len(outputs) != 48 or len({item["blinded_output_id"] for item in outputs}) != 48:
        raise TournamentGenerationError("tournament review output coverage changed")
    task_counts: dict[str, int] = {}
    for output in outputs:
        task_counts[output["task_id"]] = task_counts.get(output["task_id"], 0) + 1
        if "arm" in output or output["tokens_used"] <= 0 or not isinstance(output["branches"], list):
            raise TournamentGenerationError("tournament review packet leaked an arm or invalid usage")
    if len(task_counts) != 24 or set(task_counts.values()) != {2}:
        raise TournamentGenerationError("tournament paired task coverage changed")
    if any(review.get("claims", {}).values()) or any(receipt.get("claims", {}).values()):
        raise TournamentGenerationError("unreviewed tournament opened a claim gate")
    runtime = receipt.get("claude_runtime", {})
    credential = receipt.get("credential_activation", {})
    if (
        runtime.get("status") != "PASS_PAIRED_GENERATION"
        or runtime.get("completed_calls") != 96
        or runtime.get("authenticated_messages_api_working") is not True
        or credential.get("credential_persisted") is not False
        or credential.get("credential_value_recorded") is not False
        or receipt.get("blinding", {}).get("review_packet_content_sha256")
        != review["content_sha256"]
        or receipt.get("release_gate", {}).get("tournament_scored") is not False
    ):
        raise TournamentGenerationError("tournament runtime or blinding evidence changed")
    if any(event.get("target_reads") != 0 for event in receipt.get("chronology", [])):
        raise TournamentGenerationError("tournament generation opened a target")
    if root is not None:
        root = root.resolve()
        config = load_config(root)
        generation = _load_generation_packet(root, config)
        compatible = tuple(
            {
                **bindings,
                "generation_packet_content_sha256": generation["content_sha256"],
            }
            for bindings in _compatible_source_bindings(root)
        )
        if receipt.get("source_bindings") not in compatible:
            raise TournamentGenerationError("tournament public source binding changed")


def validate_coordinator(
    coordinator: Mapping[str, Any], review: Mapping[str, Any], public: Mapping[str, Any]
) -> None:
    body = {key: item for key, item in coordinator.items() if key != "content_sha256"}
    if (
        coordinator.get("schema_version") != COORDINATOR_SCHEMA
        or coordinator.get("content_sha256") != canonical_sha256(body)
        or coordinator.get("public_receipt_content_sha256") != public["content_sha256"]
        or coordinator.get("review_packet_content_sha256") != review["content_sha256"]
        or len(coordinator.get("mapping", [])) != 48
        or coordinator.get("claims", {}).get("safe_to_publish_before_review") is not False
    ):
        raise TournamentGenerationError("private tournament coordinator binding changed")
    key = bytes.fromhex(coordinator["unblinding_key_hex"])
    if hashlib.sha256(key).hexdigest() != public["blinding"]["unblinding_key_sha256"]:
        raise TournamentGenerationError("tournament unblinding key commitment changed")
    expected = {
        (_blinded_id(key, item["task_id"], item["arm"]), item["task_id"], item["arm"])
        for item in coordinator["mapping"]
    }
    actual = {
        (item["blinded_output_id"], item["task_id"], item["arm"])
        for item in coordinator["mapping"]
    }
    if expected != actual:
        raise TournamentGenerationError("tournament unblinding map changed")


def _private_output_path(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((root.resolve() / "work").resolve())
    except ValueError as error:
        raise TournamentGenerationError(
            "private coordinator/checkpoint output must remain under the ignored work directory"
        ) from error
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--credential-file", type=Path)
    run.add_argument("--review-output", type=Path, required=True)
    run.add_argument("--receipt-output", type=Path, required=True)
    run.add_argument("--coordinator-output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--review-packet", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        coordinator_output = _private_output_path(args.root, args.coordinator_output)
        if coordinator_output.is_file():
            existing = json.loads(coordinator_output.read_text(encoding="utf-8"))
            if existing.get("schema_version") != CHECKPOINT_SCHEMA:
                raise TournamentGenerationError(
                    "private coordinator already exists and is not a resumable checkpoint"
                )
            unblinding_key = bytes.fromhex(existing["unblinding_key_hex"])
        else:
            unblinding_key = secrets.token_bytes(32)
        review, receipt, coordinator = run_generation(
            args.root,
            unblinding_key=unblinding_key,
            credential_file=args.credential_file,
            checkpoint_path=coordinator_output,
        )
        for path, value in (
            (args.review_output, review),
            (args.receipt_output, receipt),
            (coordinator_output, coordinator),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        review = json.loads(args.review_packet.read_text(encoding="utf-8"))
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_public_generation(review, receipt, args.root)
    print(
        json.dumps(
            {
                "completed_calls": receipt["claude_runtime"]["completed_calls"],
                "content_sha256": receipt["content_sha256"],
                "review_packet_sha256": review["content_sha256"],
                "status": receipt["release_gate"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
