"""Bounded LLM-to-GPU shadow experiment for the Erdős--Straus conjecture.

The open-problem gate remains closed.  This module exercises the creative machinery on a
declared finite range, compiles well-formed suggestions into a tiny search DSL, preserves
malformed and unsuccessful ideas, and uses the existing exact sweeper as the authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

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
from .core_credential import activated_credential
from .durable_llm_attempt_journal import (
    AttemptJournalError,
    DurableAttemptJournal,
    JournaledScheduledTransport,
)
from .exponent_diophantine_sweeper import (
    _backend,
    _es_hard_members,
    _es_hard_rounds,
    _host_array,
    es_witness_is_exact,
    run_erdos_straus_sweep,
)
from .exponent_diophantine_sweeper import (
    validate_receipt as validate_sweep_receipt,
)
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/erdos_straus_creative_shadow.json"
OUTPUT_PATH = "runs/math/erdos-straus-creative-shadow/live-runtime.json"
SCHEMA_VERSION = "invariant-erdos-straus-creative-shadow-runtime-1.0"
CONFIG_SCHEMA = "invariant-erdos-straus-creative-shadow-config-1.0"
SOURCE_PATH = "src/sigma_theory_compiler/erdos_straus_creative_shadow.py"
SWEEPER_PATH = "src/sigma_theory_compiler/exponent_diophantine_sweeper.py"
JOURNAL_PATH = "work/erdos-straus-creative-shadow/llm-attempts.jsonl"

_DSL = re.compile(r"^ESDSL1\|basis=([a-z_]+)\|x=([0-9,]+)\|t=([0-9,]+)\|m=([0-9,]+)$")
_BASES = {
    "continued_fraction",
    "descent_graph",
    "divisor_pair",
    "greedy_offset",
    "lattice_transform",
    "modular_sieve",
    "polynomial_ansatz",
    "residue_cover",
}
_KNOWN_MECHANISM_TERMS = {
    "continued_fraction": ("continued fraction", "ceiling fraction"),
    "divisor_factorization": ("divisor", "factor", "b^2"),
    "greedy_egyptian_fraction": ("greedy", "egyptian"),
    "lattice_parameterization": ("lattice", "affine"),
    "modular_identity": ("modular", "residue", "congruence"),
    "polynomial_family": ("polynomial", "parametric"),
}


class ErdosStrausCreativeShadowError(ValueError):
    """Raised when the bounded experiment or its receipt fails closed."""


def _file_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _load_config(root: Path, config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    path = root / config_path
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {
        "campaign_id",
        "claim_policy",
        "claude",
        "experiment",
        "literature",
        "problem",
        "recovery",
        "schema_version",
    }:
        raise ErdosStrausCreativeShadowError("config keys changed")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ErdosStrausCreativeShadowError("config schema changed")
    policy = value["claim_policy"]
    if (
        policy.get("famous_open_problem_attempt_authorized") is not False
        or policy.get("shadow_mechanism_experiment_only") is not True
        or policy.get("level5_process_passes", 1) >= policy.get("minimum_level5_process_passes", 0)
    ):
        raise ErdosStrausCreativeShadowError("open-problem claim gate is not closed")
    experiment = value["experiment"]
    if not (
        13
        <= experiment["calibration_n_max"]
        <= experiment["benchmark_n_max"]
        <= experiment["search_n_max"]
        <= 100_000_000
    ):
        raise ErdosStrausCreativeShadowError("experiment ranges are invalid")
    try:
        creative_roles = [ClaudeRole(item) for item in experiment["creative_roles"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ErdosStrausCreativeShadowError("creative role allocation is invalid") from error
    if not creative_roles or ClaudeRole.CRITIC in creative_roles:
        raise ErdosStrausCreativeShadowError("creative role allocation is invalid")
    critic_batch_size = experiment.get("llm_critic_batch_size")
    if (
        isinstance(critic_batch_size, bool)
        or not isinstance(critic_batch_size, int)
        or not 0 <= critic_batch_size <= 16
    ):
        raise ErdosStrausCreativeShadowError("LLM critic batch size is invalid")
    if value["claude"]["maximum_calls"] != 4 or value["claude"]["maximum_total_tokens"] > 32_000:
        raise ErdosStrausCreativeShadowError("open-problem shadow budget changed")
    return value


def _public_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = config["experiment"]
    return {
        "experiment_boundary": (
            "Finite mechanism shadow only. Do not claim a proof, a new verification bound, "
            "or mathematical novelty. Every idea is retained even when malformed or falsified."
        ),
        "known_computational_baseline": (
            "For n congruent to 1 mod 12, set x=floor(n/4)+1+dx, a=4x-n, b=nx, "
            "y=ceil(b/a)+t, and accept when d=ay-b divides by exactly testing d | by."
        ),
        "machine_recipe": (
            "The expression field must be exactly ESDSL1|basis=B|x=X|t=T|m=M. "
            "B is one of continued_fraction, descent_graph, divisor_pair, greedy_offset, "
            "lattice_transform, modular_sieve, polynomial_ansatz, residue_cover. X and T "
            f"are comma-separated sets of 2 to {experiment['maximum_offsets_per_axis']} "
            f"integers from 0 to {experiment['maximum_offset']}; M is 1 to "
            f"{experiment['maximum_moduli_per_recipe']} moduli from 2 to 256."
        ),
        "problem_equation": config["problem"]["equation"],
        "problem_id": config["problem"]["id"],
        "requested_diversity": (
            "Prefer mechanisms from distant domains and different parameter scales. Label each "
            "as known_rewrite, cross_domain_synthesis, proposed_new_construction, or uncertain; "
            "the label will never be used to prune it."
        ),
    }


def _instruction(role: ClaudeRole, requested_ideas: int) -> str:
    return (
        f"Act as the {role.value} in a bounded Erdős--Straus mechanism experiment. Produce "
        f"exactly {requested_ideas} structurally different recipes spanning distant analogies, "
        "invented representations, recombinations, and independent proof strategies, not "
        "cosmetic variants. Follow machine_recipe exactly in every expression field. Put the "
        "mathematical explanation in rationale, invariants, proof_plan, known_analogues, "
        "source_idea_domains, and synthesis_note. Origin labels are fallible self-assessments "
        "and do not establish novelty. Failed ideas remain valuable."
    )


def _critic_instruction(batch_size: int) -> str:
    return (
        f"Critique exactly the {batch_size} supplied candidates in this bounded Erdős--Straus "
        "mechanism experiment. Return one typed steering action for every candidate. A reject "
        "or repair verdict must not delete an idea: identify a blocker and a concrete repair or "
        "recombination. Do not claim proof, novelty, or progress on the open conjecture."
    )


def _parse_numbers(raw: str, *, minimum: int, maximum: int, cap: int) -> tuple[int, ...] | None:
    values = tuple(sorted({int(item) for item in raw.split(",")}))
    if not 1 <= len(values) <= cap or any(item < minimum or item > maximum for item in values):
        return None
    return values


def parse_recipe(expression: str, experiment: Mapping[str, Any]) -> dict[str, Any] | None:
    match = _DSL.fullmatch(expression.strip())
    if match is None or match.group(1) not in _BASES:
        return None
    cap = int(experiment["maximum_offsets_per_axis"])
    maximum = int(experiment["maximum_offset"])
    x_offsets = _parse_numbers(match.group(2), minimum=0, maximum=maximum, cap=cap)
    t_offsets = _parse_numbers(match.group(3), minimum=0, maximum=maximum, cap=cap)
    moduli = _parse_numbers(
        match.group(4),
        minimum=2,
        maximum=256,
        cap=int(experiment["maximum_moduli_per_recipe"]),
    )
    if x_offsets is None or t_offsets is None or moduli is None:
        return None
    return {
        "basis": match.group(1),
        "moduli": list(moduli),
        "t_offsets": list(t_offsets),
        "x_offsets": list(x_offsets),
    }


def _schedule_pairs(recipe: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((int(dx), int(t)) for dx in recipe["x_offsets"] for t in recipe["t_offsets"])
    )


def _run_pairs_with_attribution(
    xp: Any, members: np.ndarray, pairs: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    started = time.perf_counter()
    n = xp.asarray(members)
    resolved = xp.zeros(n.shape, dtype=bool)
    wx = xp.zeros(n.shape, dtype=xp.int64)
    wy = xp.zeros(n.shape, dtype=xp.int64)
    winning_dx = xp.full(n.shape, -1, dtype=xp.int64)
    winning_t = xp.full(n.shape, -1, dtype=xp.int64)
    lane_tests = 0
    for dx, t in pairs:
        open_lanes = ~resolved
        open_count = int(open_lanes.sum())
        if open_count == 0:
            break
        lane_tests += open_count
        a = 3 + 4 * dx
        x = n // 4 + 1 + dx
        b = n * x
        y = (b + a - 1) // a + t
        d = a * y - b
        safe = xp.where(d > 0, d, xp.int64(1))
        ok = open_lanes & (d > 0) & (y >= x) & (((b % safe) * (y % safe)) % safe == 0)
        wx = xp.where(ok, x, wx)
        wy = xp.where(ok, y, wy)
        winning_dx = xp.where(ok, xp.int64(dx), winning_dx)
        winning_t = xp.where(ok, xp.int64(t), winning_t)
        resolved = resolved | ok
    elapsed = time.perf_counter() - started
    return (
        _host_array(wx),
        _host_array(wy),
        _host_array(resolved),
        _host_array(winning_dx),
        _host_array(winning_t),
        lane_tests,
        elapsed,
    )


def _run_pairs(
    xp: Any, members: np.ndarray, pairs: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    wx, wy, resolved, _, _, lane_tests, elapsed = _run_pairs_with_attribution(
        xp, members, pairs
    )
    return wx, wy, resolved, lane_tests, elapsed


def _witness_sample(
    members: np.ndarray, wx: np.ndarray, wy: np.ndarray, resolved: np.ndarray, limit: int = 8
) -> list[dict[str, int]]:
    indices = np.flatnonzero(resolved)
    if indices.size > limit:
        indices = indices[np.linspace(0, indices.size - 1, limit, dtype=np.int64)]
    sample = []
    for index in indices:
        n, x, y = int(members[index]), int(wx[index]), int(wy[index])
        b = n * x
        d = (4 * x - n) * y - b
        z = (b * y) // d
        if not es_witness_is_exact(n, x, y, z):
            raise ErdosStrausCreativeShadowError("creative schedule produced a false witness")
        sample.append({"n": n, "x": x, "y": y, "z": z})
    return sample


def _residue_evidence(
    members: np.ndarray, resolved: np.ndarray, moduli: Sequence[int]
) -> list[dict[str, Any]]:
    evidence = []
    for modulus in moduli:
        all_counts = np.bincount(members % modulus, minlength=modulus)
        hit_counts = np.bincount(members[resolved] % modulus, minlength=modulus)
        complete = [
            residue
            for residue in range(modulus)
            if all_counts[residue] >= 32 and hit_counts[residue] == all_counts[residue]
        ]
        evidence.append(
            {
                "complete_residues_in_finite_calibration": complete,
                "modulus": modulus,
                "observed_residues": int(np.count_nonzero(hit_counts)),
            }
        )
    return evidence


def _known_overlap(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    text = " ".join(
        [
            str(hypothesis.get("expression", "")),
            str(hypothesis.get("rationale", "")),
            " ".join(hypothesis.get("known_analogues", [])),
            " ".join(hypothesis.get("source_idea_domains", [])),
        ]
    ).lower()
    matches = sorted(
        name
        for name, terms in _KNOWN_MECHANISM_TERMS.items()
        if any(term in text for term in terms)
    )
    return {
        "automated_mechanism_matches": matches,
        "human_prior_art_review": "NOT_PERFORMED",
        "novelty_established": False,
        "status": "POSSIBLE_KNOWN_OVERLAP" if matches else "NO_COARSE_MATCH_NOT_A_NOVELTY_RESULT",
    }


def _candidate_id(role: ClaudeRole, hypothesis_id: str, ordinal: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", hypothesis_id).strip("_.-")
    return f"{role.value}.{ordinal:02d}.{clean}"[:128]


def _journal_bindings(
    root: Path, config: Mapping[str, Any], config_path: str | Path
) -> dict[str, Any]:
    return {
        "campaign_id": config["campaign_id"],
        "config_sha256": _file_sha256(root / config_path),
        "source_sha256": _file_sha256(root / SOURCE_PATH),
    }


def _open_attempt_journal(
    root: Path,
    config: Mapping[str, Any],
    config_path: str | Path,
    journal_path: str | Path,
) -> DurableAttemptJournal:
    path = Path(journal_path)
    if not path.is_absolute():
        path = root / path
    expected = _journal_bindings(root, config, config_path)
    if path.exists():
        journal = DurableAttemptJournal.load(path)
        header = journal.header
        if (
            header.get("experiment_id") != config["campaign_id"]
            or header.get("source_bindings") != expected
        ):
            raise ErdosStrausCreativeShadowError(
                "attempt journal is bound to another campaign, config, or source"
            )
        return journal
    return DurableAttemptJournal.create(
        path,
        experiment_id=config["campaign_id"],
        source_bindings=expected,
        unblinding_key=secrets.token_bytes(32),
    )


def _call_id(config: Mapping[str, Any], phase: str, ordinal: int, role: ClaudeRole) -> str:
    return f"{config['campaign_id']}:{phase}.{ordinal:02d}:{role.value}"


def _result_from_dict(value: Mapping[str, Any] | None) -> ClaudeCallResult | None:
    if value is None:
        return None
    output_raw = value.get("output")
    output = None
    if output_raw is not None:
        parseable = {key: item for key, item in output_raw.items() if key != "quarantine"}
        output = ClaudeStructuredOutput.from_mapping(parseable)
    return ClaudeCallResult(
        ClaudeCallStatus(value["status"]),
        ClaudeRole(value["role"]),
        value["benchmark_id"],
        output,
        dict(value["evidence"]),
    )


def _outcome(journal: DurableAttemptJournal, call_id: str) -> dict[str, Any] | None:
    events = [
        item
        for item in journal.events_for(call_id)
        if item["event_kind"] == "scheduled_call_outcome"
    ]
    if len(events) > 1:
        raise ErdosStrausCreativeShadowError("scheduled LLM call has multiple outcomes")
    return None if not events else dict(events[0]["payload"])


def _append_outcome(
    journal: DurableAttemptJournal,
    call_id: str,
    *,
    phase: str,
    role: ClaudeRole,
    status: str,
    result: ClaudeCallResult | None,
    errors: Sequence[str],
) -> dict[str, Any]:
    event = journal.append(
        "scheduled_call_outcome",
        call_id,
        {
            "errors": [str(item)[:1024] for item in errors],
            "phase": phase,
            "result": None if result is None else result.to_dict(),
            "role": role.value,
            "status": status,
        },
    )
    return dict(event["payload"])


def _response_events(
    journal: DurableAttemptJournal, call_id: str | None = None
) -> list[dict[str, Any]]:
    events = journal.events if call_id is None else journal.events_for(call_id)
    return [dict(item) for item in events if item["event_kind"] == "message_response"]


def _restore_budget(
    client: ClaudeCreativityClient,
    journal: DurableAttemptJournal,
    *,
    exclude_call_id: str | None = None,
) -> None:
    calls = sum(
        1
        for event in journal.events
        if event["event_kind"] == "message_dispatch"
        and event["scheduled_call_id"] != exclude_call_id
    )
    input_tokens = 0
    output_tokens = 0
    for event in _response_events(journal):
        if event["scheduled_call_id"] == exclude_call_id:
            continue
        usage = event["payload"].get("response", {}).get("usage", {})
        raw_input = usage.get("input_tokens", 0)
        raw_output = usage.get("output_tokens", 0)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (raw_input, raw_output)
        ):
            raise ErdosStrausCreativeShadowError("journaled provider usage is invalid")
        input_tokens += raw_input
        output_tokens += raw_output
    client.budget = ClaudeBudget(calls, input_tokens, output_tokens)


def _restore_model_evidence(
    client: ClaudeCreativityClient, journal: DurableAttemptJournal
) -> None:
    probes = [
        item for item in journal.events if item["event_kind"] == "model_probe_response"
    ]
    if not probes:
        raise ErdosStrausCreativeShadowError(
            "journaled response cannot replay without its model-capability probe"
        )
    recovered = []
    for probe in probes:
        payload = probe["payload"]
        response = payload.get("response", {})
        capabilities = response.get("capabilities", {})
        structured = (
            capabilities.get("structured_outputs", {})
            if isinstance(capabilities, Mapping)
            else {}
        )
        if (
            payload.get("status") != 200
            or response.get("id") != client.config.model
            or not isinstance(structured, Mapping)
            or structured.get("supported") is not True
        ):
            raise ErdosStrausCreativeShadowError(
                "journaled model-capability evidence is invalid"
            )
        recovered.append(
            {
                "capabilities_sha256": canonical_sha256(capabilities),
                "model": client.config.model,
                "structured_outputs_supported": True,
            }
        )
    if any(item != recovered[0] for item in recovered[1:]):
        raise ErdosStrausCreativeShadowError("journaled model-capability probes disagree")
    client._model_evidence = recovered[0]


class _ReplayMessageTransport:
    """Replay one durable provider response against the exact original request bytes."""

    def __init__(self, dispatch: Mapping[str, Any], response: Mapping[str, Any]) -> None:
        self.dispatch = dispatch
        self.response = response
        self.used = False

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        del headers, timeout
        if self.used or method != "POST" or body is None:
            raise ErdosStrausCreativeShadowError("journal replay attempted an unexpected request")
        expected = self.dispatch["payload"]
        if (
            expected.get("method") != method
            or expected.get("url") != url
            or expected.get("body_sha256") != hashlib.sha256(body).hexdigest()
        ):
            raise ErdosStrausCreativeShadowError(
                "journaled response request binding changed during replay"
            )
        self.used = True
        return int(self.response["payload"]["status"]), dict(
            self.response["payload"]["response"]
        )


def _run_scheduled_call(
    journal: DurableAttemptJournal,
    client: ClaudeCreativityClient,
    *,
    call_id: str,
    phase: str,
    role: ClaudeRole,
    public_payload: Mapping[str, Any],
    candidate_summaries: Sequence[Mapping[str, Any]],
    instruction: str,
    hypothesis_slots: int | None,
    base_transport: Transport,
) -> tuple[dict[str, Any], ClaudeCallResult | None]:
    existing = _outcome(journal, call_id)
    if existing is not None:
        return existing, _result_from_dict(existing.get("result"))
    previous = journal.events_for(call_id)
    dispatches = [item for item in previous if item["event_kind"] == "message_dispatch"]
    responses = [item for item in previous if item["event_kind"] == "message_response"]
    if len(dispatches) > 1 or len(responses) > 1:
        raise ErdosStrausCreativeShadowError("scheduled LLM call journal is not singular")
    if responses and not dispatches:
        raise ErdosStrausCreativeShadowError(
            "scheduled LLM response has no bound dispatch"
        )
    if dispatches and not responses:
        outcome = _append_outcome(
            journal,
            call_id,
            phase=phase,
            role=role,
            status="indeterminate_after_dispatch",
            result=None,
            errors=["process_stopped_after_dispatch_before_durable_response"],
        )
        return outcome, None
    _restore_budget(client, journal, exclude_call_id=call_id if responses else None)
    if responses:
        _restore_model_evidence(client, journal)
        client.transport = _ReplayMessageTransport(dispatches[0], responses[0])
    else:
        client.transport = JournaledScheduledTransport(
            journal,
            scheduled_call_id=call_id,
            arm="erdos_straus_shadow",
            task_id="erdos_straus_shadow",
            role=role.value,
            base_transport=base_transport,
        )
    try:
        result = client.run(
            role,
            "erdos_straus_shadow",
            public_payload,
            candidate_summaries=candidate_summaries,
            instruction_override=instruction,
            hypothesis_slots=hypothesis_slots,
        )
    except (
        ClaudeCreativityError,
        AttemptJournalError,
        ErdosStrausCreativeShadowError,
        OSError,
        TimeoutError,
    ) as error:
        outcome = _append_outcome(
            journal,
            call_id,
            phase=phase,
            role=role,
            status="client_or_contract_failure",
            result=None,
            errors=[f"{type(error).__name__}:{error!s}"],
        )
        return outcome, None
    if result.status is not ClaudeCallStatus.COMPLETED:
        return (
            {
                "errors": [f"pre_dispatch_status:{result.status.value}"],
                "phase": phase,
                "result": result.to_dict(),
                "role": role.value,
                "status": "pre_dispatch_blocked_retryable",
            },
            result,
        )
    outcome = _append_outcome(
        journal,
        call_id,
        phase=phase,
        role=role,
        status="completed",
        result=result,
        errors=[],
    )
    return outcome, result


def _candidate_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    hypothesis = item["hypothesis"]
    return {
        "candidate_id": item["idea_id"],
        "expression": hypothesis["expression"],
        "known_analogues": hypothesis["known_analogues"],
        "llm_origin_assessment": hypothesis["llm_origin_assessment"],
        "representation": hypothesis["representation"],
        "source_idea_domains": hypothesis["source_idea_domains"],
    }


def _creative_calls(
    config: Mapping[str, Any],
    journal: DurableAttemptJournal,
    *,
    base_transport: Transport = urllib_transport,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    claude = ClaudeCreativityClient(ClaudeAPIConfig.from_mapping(config["claude"]))
    base_payload = _public_payload(config)
    calls = []
    attempts = []
    ideas = []
    roles = [ClaudeRole(item) for item in config["experiment"]["creative_roles"]]
    requested = int(config["experiment"]["requested_ideas_per_creative_call"])
    for call_ordinal, role in enumerate(roles, 1):
        payload = dict(base_payload)
        if role is ClaudeRole.RECOMBINER:
            payload["parent_palette"] = [
                {
                    "expression": item["hypothesis"]["expression"],
                    "idea_id": item["idea_id"],
                    "origin": item["hypothesis"]["llm_origin_assessment"],
                    "source_domains": item["hypothesis"]["source_idea_domains"],
                }
                for item in ideas
            ]
        outcome, result = _run_scheduled_call(
            journal,
            claude,
            call_id=_call_id(config, "creative", call_ordinal, role),
            phase="creative",
            role=role,
            public_payload=payload,
            candidate_summaries=(),
            instruction=_instruction(role, requested),
            hypothesis_slots=None,
            base_transport=base_transport,
        )
        attempts.append(outcome)
        if (
            result is None
            or result.status is not ClaudeCallStatus.COMPLETED
            or result.output is None
        ):
            continue
        record = result.to_dict()
        calls.append(record)
        for ordinal, hypothesis in enumerate(record["output"]["hypotheses"], 1):
            ideas.append(
                {
                    "hypothesis": hypothesis,
                    "idea_id": _candidate_id(role, hypothesis["hypothesis_id"], ordinal),
                    "role": role.value,
                }
            )
    if not ideas:
        raise ErdosStrausCreativeShadowError("no journaled Claude idea call completed")
    critic_actions: dict[str, dict[str, Any]] = {}
    critic_batch_size = int(config["experiment"]["llm_critic_batch_size"])
    if critic_batch_size:
        batches = [
            ideas[index : index + critic_batch_size]
            for index in range(0, len(ideas), critic_batch_size)
        ]
        for batch_ordinal, batch in enumerate(batches, 1):
            summaries = [_candidate_summary(item) for item in batch]
            outcome, result = _run_scheduled_call(
                journal,
                claude,
                call_id=_call_id(config, "critic", batch_ordinal, ClaudeRole.CRITIC),
                phase="critic",
                role=ClaudeRole.CRITIC,
                public_payload=base_payload,
                candidate_summaries=summaries,
                instruction=_critic_instruction(len(batch)),
                hypothesis_slots=None,
                base_transport=base_transport,
            )
            attempts.append(outcome)
            if result is None or result.output is None:
                continue
            calls.append(result.to_dict())
            for action in result.output.steering_actions:
                critic_actions[action.candidate_id] = action.to_dict()
    for item in ideas:
        executable = (
            parse_recipe(item["hypothesis"]["expression"], config["experiment"]) is not None
        )
        deterministic = {
            "blocker_kind": "none" if executable else "machine_dsl_parse_failure",
            "candidate_id": item["idea_id"],
            "distance_denominator": 1,
            "distance_numerator": 0 if executable else 1,
            "repair": (
                "Proceed to exact finite execution; correctness and novelty remain unverified."
                if executable
                else "Retain the idea text and translate it into a valid ESDSL1 recipe."
            ),
            "verdict": "retain" if executable else "repair",
        }
        item["critic"] = critic_actions.get(item["idea_id"], deterministic)
        item["critic_source"] = (
            "journaled_llm_critic_retained_without_pruning"
            if item["idea_id"] in critic_actions
            else "deterministic_machine_admission_not_llm_critique"
        )
    _restore_budget(claude, journal)
    return ideas, {
        "budget": claude.budget.to_dict(),
        "calls": calls,
        "attempt_journal": {
            "content_sha256": journal.content_sha256,
            "credential_values_persisted": False,
            "event_counts": journal.event_counts(),
            "private_path_persisted_in_receipt": False,
            "response_before_validation": True,
            "resume_redispatches_completed_slots": False,
        },
        "attempts": attempts,
        "critic_kind": (
            "journaled_llm_batches_with_deterministic_fallback_no_pruning"
            if critic_batch_size
            else "deterministic_machine_admission_not_llm_critique"
        ),
    }


def _evaluate_ideas(
    ideas: list[dict[str, Any]], config: Mapping[str, Any], xp: Any, device: str
) -> dict[str, Any]:
    experiment = config["experiment"]
    members = _es_hard_members(int(experiment["calibration_n_max"]))
    executable = 0
    all_pairs: set[tuple[int, int]] = set()
    total_lane_tests = 0
    for item in ideas:
        hypothesis = item["hypothesis"]
        recipe = parse_recipe(hypothesis["expression"], experiment)
        item["llm_self_assessed_origin"] = hypothesis["llm_origin_assessment"]
        item["prior_art"] = _known_overlap(hypothesis)
        if recipe is None:
            item["execution"] = {
                "admission": "RETAINED_NONEXECUTABLE",
                "reason": "expression_did_not_compile_to_esdsl1",
            }
            continue
        executable += 1
        pairs = _schedule_pairs(recipe)
        all_pairs.update(pairs)
        wx, wy, resolved, lane_tests, elapsed = _run_pairs(xp, members, pairs)
        total_lane_tests += lane_tests
        item["execution"] = {
            "admission": "EXECUTED_EXACT_MODULAR_SCREEN",
            "device": device,
            "direct_parameter_pairs": [list(pair) for pair in pairs],
            "elapsed_seconds": f"{elapsed:.6f}",
            "exact_lane_tests": lane_tests,
            "finite_calibration_hard_cases": int(members.size),
            "recipe": recipe,
            "residue_evidence": _residue_evidence(members, resolved, recipe["moduli"]),
            "resolved_hard_cases": int(resolved.sum()),
            "witness_sample": _witness_sample(members, wx, wy, resolved),
        }
    return {
        "direct_unique_parameter_pairs": len(all_pairs),
        "executable_ideas": executable,
        "ideas": ideas,
        "nonexecutable_ideas_retained": len(ideas) - executable,
        "parameter_pairs": sorted([list(pair) for pair in all_pairs]),
        "total_idea_lane_tests": total_lane_tests,
    }


def _mutated_pairs(
    direct_pairs: Sequence[Sequence[int]], experiment: Mapping[str, Any]
) -> tuple[tuple[int, int], ...]:
    radius = int(experiment["mutation_radius"])
    maximum = int(experiment["maximum_offset"])
    result = set()
    for raw_dx, raw_t in direct_pairs:
        for delta_x in range(-radius, radius + 1):
            for delta_t in range(-radius, radius + 1):
                dx, t = int(raw_dx) + delta_x, int(raw_t) + delta_t
                if 0 <= dx <= maximum and 0 <= t <= maximum:
                    result.add((dx, t))
    return tuple(sorted(result, key=lambda item: (item[0] + item[1], item)))


def _mutation_lineage(
    creative: Mapping[str, Any], experiment: Mapping[str, Any]
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    radius = int(experiment["mutation_radius"])
    maximum = int(experiment["maximum_offset"])
    lineage: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for idea in creative["ideas"]:
        execution = idea["execution"]
        if execution["admission"] != "EXECUTED_EXACT_MODULAR_SCREEN":
            continue
        for raw_dx, raw_t in execution["direct_parameter_pairs"]:
            direct_dx, direct_t = int(raw_dx), int(raw_t)
            for delta_x in range(-radius, radius + 1):
                for delta_t in range(-radius, radius + 1):
                    dx, t = direct_dx + delta_x, direct_t + delta_t
                    if not (0 <= dx <= maximum and 0 <= t <= maximum):
                        continue
                    lineage.setdefault((dx, t), []).append(
                        {
                            "basis": execution["recipe"]["basis"],
                            "direct_pair": [direct_dx, direct_t],
                            "idea_id": idea["idea_id"],
                            "llm_self_assessed_origin": idea["llm_self_assessed_origin"],
                            "mutation_delta": [delta_x, delta_t],
                            "role": idea["role"],
                        }
                    )
    return {
        pair: sorted(records, key=canonical_sha256)
        for pair, records in sorted(
            lineage.items(), key=lambda item: (sum(item[0]), item[0])
        )
    }


def _summarize_lineage_attribution(
    records: Sequence[Mapping[str, Any]], creative: Mapping[str, Any]
) -> dict[str, Any]:
    idea_hits: dict[str, set[int]] = {}
    idea_pairs: dict[str, set[tuple[int, int]]] = {}
    basis_hits: dict[str, set[int]] = {}
    basis_pairs: dict[str, set[tuple[int, int]]] = {}
    winning_pair_hits: dict[tuple[int, int], int] = {}
    for record in records:
        n = int(record["n"])
        winning_pair = tuple(int(item) for item in record["winning_pair"])
        winning_pair_hits[winning_pair] = winning_pair_hits.get(winning_pair, 0) + 1
        for parent in record["parent_lineages"]:
            idea_id = parent["idea_id"]
            basis = parent["basis"]
            idea_hits.setdefault(idea_id, set()).add(n)
            idea_pairs.setdefault(idea_id, set()).add(winning_pair)
            basis_hits.setdefault(basis, set()).add(n)
            basis_pairs.setdefault(basis, set()).add(winning_pair)
    idea_metadata = {
        item["idea_id"]: {
            "basis": item["execution"].get("recipe", {}).get("basis"),
            "llm_self_assessed_origin": item["llm_self_assessed_origin"],
            "role": item["role"],
        }
        for item in creative["ideas"]
    }
    return {
        "all_resolved_hits_have_parent_lineage": all(
            bool(item["parent_lineages"]) for item in records
        ),
        "attribution_rule": "first_exact_success_in_canonical_mutated_pair_order",
        "basis_linked_hits": [
            {
                "basis": basis,
                "linked_resolved_hits": len(basis_hits[basis]),
                "winning_pairs": [list(pair) for pair in sorted(basis_pairs[basis])],
            }
            for basis in sorted(basis_hits)
        ],
        "causal_or_novelty_credit_claimed": False,
        "idea_linked_hits": [
            {
                "basis": idea_metadata[idea_id]["basis"],
                "idea_id": idea_id,
                "linked_resolved_hits": len(idea_hits[idea_id]),
                "llm_self_assessed_origin": idea_metadata[idea_id][
                    "llm_self_assessed_origin"
                ],
                "role": idea_metadata[idea_id]["role"],
                "winning_pairs": [list(pair) for pair in sorted(idea_pairs[idea_id])],
            }
            for idea_id in sorted(idea_hits)
        ],
        "multi_basis_lineage_hits": sum(
            len({parent["basis"] for parent in item["parent_lineages"]}) > 1
            for item in records
        ),
        "multi_idea_lineage_hits": sum(
            len({parent["idea_id"] for parent in item["parent_lineages"]}) > 1
            for item in records
        ),
        "multi_parent_lineage_hits": sum(
            len(item["parent_lineages"]) > 1 for item in records
        ),
        "resolved_hit_records": [dict(item) for item in records],
        "resolved_hit_records_sha256": canonical_sha256(records),
        "winning_pair_hit_counts": [
            {"resolved_hits": winning_pair_hits[pair], "winning_pair": list(pair)}
            for pair in sorted(winning_pair_hits)
        ],
    }


def _hard_tail_experiment(
    experiment: Mapping[str, Any], creative: Mapping[str, Any], xp: Any, device: str
) -> dict[str, Any]:
    n_max = int(experiment["search_n_max"])
    members = _es_hard_members(n_max)
    baseline_started = time.perf_counter()
    _, _, baseline_resolved, baseline_lane_tests = _es_hard_rounds(
        xp,
        members,
        int(experiment["baseline_x_rounds"]),
        int(experiment["baseline_t_rounds"]),
    )
    baseline_elapsed = time.perf_counter() - baseline_started
    tail = members[~baseline_resolved]
    direct = tuple(tuple(item) for item in creative["parameter_pairs"])
    lineage_by_pair = _mutation_lineage(creative, experiment)
    mutated = tuple(lineage_by_pair)
    if mutated != _mutated_pairs(creative["parameter_pairs"], experiment):
        raise ErdosStrausCreativeShadowError(
            "mutation lineage does not cover the exact union search schedule"
        )
    baseline_pairs = {
        (dx, t)
        for dx in range(int(experiment["baseline_x_rounds"]))
        for t in range(int(experiment["baseline_t_rounds"]))
    }
    novel_pairs = tuple(pair for pair in mutated if pair not in baseline_pairs)
    (
        wx,
        wy,
        resolved,
        winning_dx,
        winning_t,
        creative_lane_tests,
        creative_elapsed,
    ) = _run_pairs_with_attribution(xp, tail, novel_pairs)
    verified_witnesses = []
    attribution_records = []
    pair_order = {pair: ordinal for ordinal, pair in enumerate(novel_pairs)}
    for index in np.flatnonzero(resolved):
        n, x, y = int(tail[index]), int(wx[index]), int(wy[index])
        winning_pair = (int(winning_dx[index]), int(winning_t[index]))
        parent_lineages = lineage_by_pair.get(winning_pair, [])
        if not parent_lineages or winning_pair not in pair_order:
            raise ErdosStrausCreativeShadowError(
                "creative tail hit has no exact parent mutation lineage"
            )
        b = n * x
        d = (4 * x - n) * y - b
        z = (b * y) // d
        if not es_witness_is_exact(n, x, y, z):
            raise ErdosStrausCreativeShadowError(
                "creative tail GPU hit failed independent CPU exact verification"
            )
        witness = {"n": n, "x": x, "y": y, "z": z}
        verified_witnesses.append(witness)
        attribution_records.append(
            {
                "canonical_pair_order_ordinal": pair_order[winning_pair],
                "n": n,
                "parent_lineages": parent_lineages,
                "winning_pair": list(winning_pair),
                "witness": witness,
            }
        )
    attribution = _summarize_lineage_attribution(attribution_records, creative)
    attribution["all_resolved_hits_have_parent_lineage"] = (
        attribution["all_resolved_hits_have_parent_lineage"]
        and len(attribution_records) == int(resolved.sum())
    )
    uniform_universe = sorted(
        (dx, t)
        for dx in range(int(experiment["maximum_offset"]) + 1)
        for t in range(int(experiment["maximum_offset"]) + 1)
        if (dx, t) not in baseline_pairs
    )
    rng = np.random.default_rng(int(experiment["matched_control_seed"]))
    creative_count = int(resolved.sum())

    def summarize_controls(
        label: str, control_counts: Sequence[int], control_lane_tests: Sequence[int]
    ) -> dict[str, Any]:
        random_at_least_creative = sum(count >= creative_count for count in control_counts)
        return {
            "control_kind": label,
            "control_resolved_counts": control_counts,
            "creative_outperformed_random_median": creative_count
            > float(np.median(control_counts)),
            "creative_percentile": (
                f"{sum(count <= creative_count for count in control_counts) / len(control_counts):.6f}"
            ),
            "empirical_one_sided_p": (
                f"{(random_at_least_creative + 1) / (len(control_counts) + 1):.6f}"
            ),
            "maximum_resolved": max(control_counts),
            "mean_resolved": f"{float(np.mean(control_counts)):.6f}",
            "median_resolved": f"{float(np.median(control_counts)):.6f}",
            "minimum_resolved": min(control_counts),
            "random_at_least_creative": random_at_least_creative,
            "total_exact_lane_tests": sum(control_lane_tests),
            "trials": len(control_counts),
        }

    def run_controls(universe: Sequence[tuple[int, int]], label: str) -> dict[str, Any]:
        control_counts = []
        control_lane_tests = []
        for _ in range(int(experiment["matched_control_trials"])):
            indices = rng.choice(len(universe), size=len(novel_pairs), replace=False)
            control_pairs = tuple(universe[int(index)] for index in indices)
            _, _, control_resolved, lane_tests, _ = _run_pairs(xp, tail, control_pairs)
            control_counts.append(int(control_resolved.sum()))
            control_lane_tests.append(lane_tests)
        return summarize_controls(label, control_counts, control_lane_tests)

    def rewire_pairs() -> tuple[tuple[tuple[int, int], ...], int]:
        edges = list(novel_pairs)
        edge_set = set(edges)
        successful = 0
        for _ in range(20 * len(edges)):
            first_index, second_index = rng.choice(len(edges), size=2, replace=False)
            first = edges[int(first_index)]
            second = edges[int(second_index)]
            if first[0] == second[0] or first[1] == second[1]:
                continue
            replacement_first = (first[0], second[1])
            replacement_second = (second[0], first[1])
            if (
                replacement_first in edge_set
                or replacement_second in edge_set
                or replacement_first in baseline_pairs
                or replacement_second in baseline_pairs
            ):
                continue
            edge_set.remove(first)
            edge_set.remove(second)
            edge_set.add(replacement_first)
            edge_set.add(replacement_second)
            edges[int(first_index)] = replacement_first
            edges[int(second_index)] = replacement_second
            successful += 1
        return tuple(edges), successful

    x_support = sorted({dx for dx, _ in novel_pairs})
    t_support = sorted({t for _, t in novel_pairs})
    support_universe = sorted(
        (dx, t) for dx in x_support for t in t_support if (dx, t) not in baseline_pairs
    )
    uniform_controls = run_controls(uniform_universe, "uniform_full_declared_domain")
    support_controls = run_controls(support_universe, "llm_offset_support_cross_product")
    rewired_counts = []
    rewired_lane_tests = []
    rewired_swaps = []
    for _ in range(int(experiment["matched_control_trials"])):
        rewired_pairs, successful_swaps = rewire_pairs()
        _, _, rewired_resolved, lane_tests, _ = _run_pairs(xp, tail, rewired_pairs)
        rewired_counts.append(int(rewired_resolved.sum()))
        rewired_lane_tests.append(lane_tests)
        rewired_swaps.append(successful_swaps)
    rewired_controls = summarize_controls(
        "degree_preserving_pair_rewire", rewired_counts, rewired_lane_tests
    )
    rewired_controls["exact_x_marginal_frequency_matched"] = True
    rewired_controls["exact_t_marginal_frequency_matched"] = True
    rewired_controls["minimum_successful_edge_swaps"] = min(rewired_swaps)
    matched_controls = {
        "candidate_pair_count_matched": True,
        "early_stop_rule_matched": True,
        "llm_x_support": x_support,
        "llm_t_support": t_support,
        "parameter_domain_matched": True,
        "pairing_only_rewire": rewired_controls,
        "seed": int(experiment["matched_control_seed"]),
        "support_matched": support_controls,
        "total_exact_lane_tests": (
            uniform_controls["total_exact_lane_tests"]
            + support_controls["total_exact_lane_tests"]
            + rewired_controls["total_exact_lane_tests"]
        ),
        "uniform_domain": uniform_controls,
        "wall_clock_budget_claimed_matched": False,
    }
    return {
        "baseline": {
            "device": device,
            "elapsed_seconds": f"{baseline_elapsed:.6f}",
            "exact_lane_tests": baseline_lane_tests,
            "hard_cases": int(members.size),
            "resolved": int(baseline_resolved.sum()),
            "unresolved_tail": int(tail.size),
        },
        "creative_tail": {
            "direct_parameter_pairs": len(direct),
            "device": device,
            "elapsed_seconds": f"{creative_elapsed:.6f}",
            "exact_lane_tests": creative_lane_tests,
            "independent_cpu_exact_verified": len(verified_witnesses),
            "integer_gpu_screen_cpu_identity_agreement": len(verified_witnesses)
            == int(resolved.sum()),
            "mutated_parameter_pairs": len(mutated),
            "pairs_outside_fixed_baseline": len(novel_pairs),
            "parent_lineage_attribution": attribution,
            "resolved_from_baseline_tail": int(resolved.sum()),
            "still_requiring_complete_divisor_search": int(tail.size - resolved.sum()),
            "verified_witness_corpus_sha256": canonical_sha256(verified_witnesses),
            "witness_sample": _witness_sample(tail, wx, wy, resolved),
        },
        "matched_random_controls": matched_controls,
    }


def _benchmark(config: Mapping[str, Any]) -> dict[str, Any]:
    n_max = int(config["experiment"]["benchmark_n_max"])
    cpu = run_erdos_straus_sweep(n_max, use_gpu=False)
    gpu = run_erdos_straus_sweep(n_max, use_gpu=True)
    stable_match = (
        cpu["decision"] == gpu["decision"]
        and cpu["prefilter"]["gpu_lane_tests"] == gpu["prefilter"]["gpu_lane_tests"]
        and cpu["results"]["unsolvable_candidates"] == gpu["results"]["unsolvable_candidates"]
    )
    if not stable_match:
        raise ErdosStrausCreativeShadowError("CPU/GPU control disagreement")
    return {
        "cpu": {
            "device": cpu["device"],
            "elapsed_seconds": cpu["elapsed_seconds"],
            "throughput_per_second": cpu["throughput_per_second"],
        },
        "exact_result_match": True,
        "gpu": {
            "device": gpu["device"],
            "elapsed_seconds": gpu["elapsed_seconds"],
            "throughput_per_second": gpu["throughput_per_second"],
        },
        "n_max": n_max,
        "speedup": f"{float(cpu['elapsed_seconds']) / float(gpu['elapsed_seconds']):.6f}",
    }


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    config_path: str | Path,
    ideas: list[dict[str, Any]],
    claude: Mapping[str, Any],
    credential_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    root = root.resolve()
    xp, device = _backend(True)
    if device == "cpu-numpy":
        raise ErdosStrausCreativeShadowError("creative shadow build requires a CUDA device")
    creative = _evaluate_ideas(ideas, config, xp, device)
    hard_tail = _hard_tail_experiment(config["experiment"], creative, xp, device)
    benchmark = _benchmark(config)
    sweep = run_erdos_straus_sweep(int(config["experiment"]["search_n_max"]), use_gpu=True)
    validate_sweep_receipt(sweep)
    origin_counts = Counter(item["llm_self_assessed_origin"] for item in creative["ideas"])
    critic_counts = Counter(item["critic"]["verdict"] for item in creative["ideas"])
    body: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "claim_boundary": {
            "finite_search_decides_conjecture": False,
            "famous_open_problem_attempt_authorized": False,
            "human_prior_art_review_complete": False,
            "llm_self_assessment_establishes_novelty": False,
            "new_verification_bound_claimed": False,
            "open_problem_solved": False,
        },
        "claude": {
            **claude,
            "completed_calls": len(claude["calls"]),
            "critic_verdict_counts": dict(sorted(critic_counts.items())),
            "credential_activation": credential_evidence,
            "origin_counts": dict(sorted(origin_counts.items())),
            "proposed_ideas": len(creative["ideas"]),
            "requested_ideas_per_creative_call": config["experiment"][
                "requested_ideas_per_creative_call"
            ],
        },
        "config": {
            "config_sha256": _file_sha256(root / config_path),
            "source_sha256": _file_sha256(root / SOURCE_PATH),
            "sweeper_sha256": _file_sha256(root / SWEEPER_PATH),
        },
        "creative_search": creative,
        "finite_sweep": sweep,
        "gpu_benchmark": benchmark,
        "hard_tail_funnel": hard_tail,
        "literature": config["literature"],
        "problem": config["problem"],
        "recovery": config["recovery"],
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_BOUNDED_CREATIVE_SHADOW_NO_OPEN_PROBLEM_CLAIM",
    }
    body["accounting"] = {
        "baseline_gpu_lane_tests": hard_tail["baseline"]["exact_lane_tests"],
        "creative_tail_lane_tests": hard_tail["creative_tail"]["exact_lane_tests"],
        "denominators_covered": sweep["results"]["coverage"]["class_total"],
        "executable_llm_ideas": creative["executable_ideas"],
        "llm_ideas_proposed": len(creative["ideas"]),
        "llm_provider_calls": (
            claude["budget"]["calls"]
            + sum(
                item["completed_provider_calls"]
                for item in config["recovery"]["discarded_attempts"]
            )
        ),
        "retained_llm_provider_calls": claude["budget"]["calls"],
        "mutated_parameter_pairs": hard_tail["creative_tail"]["mutated_parameter_pairs"],
        "matched_control_lane_tests": hard_tail["matched_random_controls"][
            "total_exact_lane_tests"
        ],
        "total_exact_modular_lane_tests": (
            creative["total_idea_lane_tests"]
            + hard_tail["baseline"]["exact_lane_tests"]
            + hard_tail["creative_tail"]["exact_lane_tests"]
            + hard_tail["matched_random_controls"]["total_exact_lane_tests"]
        ),
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root, config_path=config_path)
    return body


def run_live(
    root: Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    journal_path: str | Path = JOURNAL_PATH,
    base_transport: Transport = urllib_transport,
) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root, config_path)
    journal = _open_attempt_journal(root, config, config_path, journal_path)
    with activated_credential(
        project_root=root,
        env_var=config["claude"]["credential_env_var"],
    ) as activation:
        ideas, claude = _creative_calls(config, journal, base_transport=base_transport)
        credential_evidence = activation.to_evidence()
    return _build_receipt(root, config, config_path, ideas, claude, credential_evidence)


def rebind_receipt(
    root: Path, previous: Mapping[str, Any], *, config_path: str | Path = CONFIG_PATH
) -> dict[str, Any]:
    """Rebuild deterministic evidence and source seals without another provider call."""

    root = root.resolve()
    config = _load_config(root, config_path)
    prior_claude = previous.get("claude", {})
    prior_ideas = previous.get("creative_search", {}).get("ideas", [])
    if not isinstance(prior_ideas, list) or not prior_ideas:
        raise ErdosStrausCreativeShadowError("previous receipt has no reusable ideas")
    ideas = [
        {
            "critic": dict(item["critic"]),
            "critic_source": item.get(
                "critic_source", "deterministic_machine_admission_not_llm_critique"
            ),
            "hypothesis": dict(item["hypothesis"]),
            "idea_id": item["idea_id"],
            "role": item["role"],
        }
        for item in prior_ideas
    ]
    claude = {
        "budget": dict(prior_claude["budget"]),
        "calls": list(prior_claude["calls"]),
        "critic_kind": prior_claude["critic_kind"],
    }
    for optional in ("attempt_journal", "attempts"):
        if optional in prior_claude:
            claude[optional] = prior_claude[optional]
    return _build_receipt(
        root,
        config,
        config_path,
        ideas,
        claude,
        dict(prior_claude["credential_activation"]),
    )


def validate_receipt(
    value: Mapping[str, Any], root: Path, *, config_path: str | Path = CONFIG_PATH
) -> None:
    root = root.resolve()
    config = _load_config(root, config_path)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("status") != (
        "PASS_BOUNDED_CREATIVE_SHADOW_NO_OPEN_PROBLEM_CLAIM"
    ):
        raise ErdosStrausCreativeShadowError("receipt identity changed")
    if canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    ) != value.get("content_sha256"):
        raise ErdosStrausCreativeShadowError("receipt content seal failed")
    if value.get("config") != {
        "config_sha256": _file_sha256(root / config_path),
        "source_sha256": _file_sha256(root / SOURCE_PATH),
        "sweeper_sha256": _file_sha256(root / SWEEPER_PATH),
    }:
        raise ErdosStrausCreativeShadowError("receipt source bindings changed")
    boundary = value["claim_boundary"]
    if any(boundary.values()):
        raise ErdosStrausCreativeShadowError("a prohibited claim is true")
    claude = value["claude"]
    completed_calls = claude["completed_calls"]
    budget_calls = claude["budget"]["calls"]
    if (
        completed_calls != len(claude["calls"])
        or not 1 <= completed_calls <= budget_calls <= config["claude"]["maximum_calls"]
    ):
        raise ErdosStrausCreativeShadowError("Claude call accounting changed")
    proposed = claude["proposed_ideas"]
    if not 1 <= proposed <= 16 * len(config["experiment"]["creative_roles"]):
        raise ErdosStrausCreativeShadowError("idea count is outside the structured-output cap")
    if (
        claude["requested_ideas_per_creative_call"]
        != config["experiment"]["requested_ideas_per_creative_call"]
    ):
        raise ErdosStrausCreativeShadowError("requested idea allocation changed")
    if claude["credential_activation"].get("credential_persisted") is not False:
        raise ErdosStrausCreativeShadowError("credential persistence boundary changed")
    journal = claude.get("attempt_journal")
    if journal is not None and (
        journal.get("credential_values_persisted") is not False
        or journal.get("private_path_persisted_in_receipt") is not False
        or journal.get("response_before_validation") is not True
        or journal.get("resume_redispatches_completed_slots") is not False
        or journal.get("event_counts", {}).get("message_dispatch") != budget_calls
    ):
        raise ErdosStrausCreativeShadowError("durable attempt journal evidence changed")
    validate_sweep_receipt(value["finite_sweep"])
    direct_union: set[tuple[int, int]] = set()
    for idea in value["creative_search"]["ideas"]:
        if idea["execution"]["admission"] == "EXECUTED_EXACT_MODULAR_SCREEN":
            recipe = parse_recipe(idea["hypothesis"]["expression"], config["experiment"])
            expected_pairs = _schedule_pairs(recipe or {}) if recipe is not None else ()
            if (
                recipe != idea["execution"]["recipe"]
                or [list(pair) for pair in expected_pairs]
                != idea["execution"]["direct_parameter_pairs"]
            ):
                raise ErdosStrausCreativeShadowError(
                    "idea direct schedule or typed recipe changed"
                )
            direct_union.update(expected_pairs)
            for witness in idea["execution"]["witness_sample"]:
                if not es_witness_is_exact(**witness):
                    raise ErdosStrausCreativeShadowError("idea witness failed exact replay")
        if idea["prior_art"]["novelty_established"] is not False:
            raise ErdosStrausCreativeShadowError("automated prior art claimed novelty")
    for witness in value["hard_tail_funnel"]["creative_tail"]["witness_sample"]:
        if not es_witness_is_exact(**witness):
            raise ErdosStrausCreativeShadowError("tail witness failed exact replay")
    tail = value["hard_tail_funnel"]["creative_tail"]
    if (
        tail["independent_cpu_exact_verified"] != tail["resolved_from_baseline_tail"]
        or tail["integer_gpu_screen_cpu_identity_agreement"] is not True
    ):
        raise ErdosStrausCreativeShadowError("creative tail independent verification changed")
    creative = value["creative_search"]
    if [list(pair) for pair in sorted(direct_union)] != creative["parameter_pairs"]:
        raise ErdosStrausCreativeShadowError("creative direct-pair union changed")
    lineage_by_pair = _mutation_lineage(creative, config["experiment"])
    baseline_pairs = {
        (dx, t)
        for dx in range(int(config["experiment"]["baseline_x_rounds"]))
        for t in range(int(config["experiment"]["baseline_t_rounds"]))
    }
    novel_pairs = tuple(pair for pair in lineage_by_pair if pair not in baseline_pairs)
    pair_order = {pair: ordinal for ordinal, pair in enumerate(novel_pairs)}
    attribution = tail.get("parent_lineage_attribution", {})
    records = attribution.get("resolved_hit_records", [])
    if (
        not isinstance(records, list)
        or len(records) != tail["resolved_from_baseline_tail"]
        or len({item.get("n") for item in records}) != len(records)
    ):
        raise ErdosStrausCreativeShadowError("creative lineage record coverage changed")
    for record in records:
        pair = tuple(record["winning_pair"])
        witness = record["witness"]
        if (
            pair not in pair_order
            or record["canonical_pair_order_ordinal"] != pair_order[pair]
            or record["parent_lineages"] != lineage_by_pair[pair]
            or record["n"] != witness["n"]
            or not es_witness_is_exact(**witness)
        ):
            raise ErdosStrausCreativeShadowError("creative lineage record failed exact replay")
        n = int(record["n"])
        dx, t = (int(item) for item in pair)
        x = n // 4 + 1 + dx
        a = 4 * x - n
        b = n * x
        y = (b + a - 1) // a + t
        d = a * y - b
        z = (b * y) // d if d > 0 and (b * y) % d == 0 else 0
        if witness != {"n": n, "x": x, "y": y, "z": z}:
            raise ErdosStrausCreativeShadowError(
                "creative lineage winning pair does not reproduce its witness"
            )
    expected_attribution = _summarize_lineage_attribution(records, creative)
    expected_attribution["all_resolved_hits_have_parent_lineage"] = (
        expected_attribution["all_resolved_hits_have_parent_lineage"]
        and len(records) == tail["resolved_from_baseline_tail"]
    )
    if attribution != expected_attribution:
        raise ErdosStrausCreativeShadowError("creative lineage aggregate changed")
    accounting = value["accounting"]
    if accounting["llm_ideas_proposed"] != value["claude"]["proposed_ideas"]:
        raise ErdosStrausCreativeShadowError("idea accounting mismatch")
    discarded_calls = sum(
        item["completed_provider_calls"] for item in config["recovery"]["discarded_attempts"]
    )
    if (
        accounting["llm_provider_calls"] != budget_calls + discarded_calls
        or accounting["retained_llm_provider_calls"] != budget_calls
    ):
        raise ErdosStrausCreativeShadowError("provider recovery accounting mismatch")
    if accounting["denominators_covered"] != config["experiment"]["search_n_max"] - 1:
        raise ErdosStrausCreativeShadowError("finite coverage accounting mismatch")
    controls = value["hard_tail_funnel"]["matched_random_controls"]
    if (
        controls["uniform_domain"]["trials"] != config["experiment"]["matched_control_trials"]
        or controls["support_matched"]["trials"] != config["experiment"]["matched_control_trials"]
        or controls["pairing_only_rewire"]["trials"]
        != config["experiment"]["matched_control_trials"]
        or controls["pairing_only_rewire"]["exact_x_marginal_frequency_matched"] is not True
        or controls["pairing_only_rewire"]["exact_t_marginal_frequency_matched"] is not True
        or controls["candidate_pair_count_matched"] is not True
        or controls["parameter_domain_matched"] is not True
        or controls["early_stop_rule_matched"] is not True
    ):
        raise ErdosStrausCreativeShadowError("matched random control contract changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-live", "rebind", "validate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--journal", default=JOURNAL_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / OUTPUT_PATH
    if args.command == "run-live":
        receipt = run_live(root, config_path=args.config, journal_path=args.journal)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "rebind":
        previous = json.loads(output.read_text(encoding="utf-8"))
        receipt = rebind_receipt(root, previous, config_path=args.config)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        receipt = json.loads(output.read_text(encoding="utf-8"))
        validate_receipt(receipt, root, config_path=args.config)
    print(
        json.dumps(
            {
                "accounting": receipt["accounting"],
                "content_sha256": receipt["content_sha256"],
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
