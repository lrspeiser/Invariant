"""Live counterexample-guided repair and recombination of retained piecewise ideas.

The campaign gives Claude only anonymous fresh training rows, retained parent lineage, and
counterexamples that were already opened by the earlier replay.  Fresh rotation holdouts are not
included in any provider payload.  Every returned idea remains in lineage regardless of origin
self-label or executor admission.  Admitted descendants are replayed by the primary and independent
exact evaluators and compared with a resource-matched random control.  Model participation,
execution, fit, behavioral novelty, proof-mechanism novelty, literature novelty, and proof are
separate claims throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .claude_creativity_api import (
    ClaudeAPIConfig,
    ClaudeCallStatus,
    ClaudeCreativityClient,
    ClaudeCreativityError,
    ClaudeHypothesis,
    ClaudeRole,
    Transport,
    _structured_output_schema,
    urllib_transport,
)
from .core_credential import activated_credential
from .external_claude_transport import ProviderCompatibleClaudeTransport
from .external_creativity_validation import (
    EXECUTABLE_PROPOSER_INSTRUCTION,
    Benchmark,
    ExternalSource,
    Observation,
    SealedTarget,
    Variable,
    _behavior,
    _candidate_resource_profile,
    _claude_candidate,
    _fraction_text,
    _loss,
    _proof_plan_search,
    independently_predict,
    load_public_benchmarks,
    predict,
    random_controls,
    unseal_targets,
)
from .idea_lineage import validate_idea_archive
from .retained_piecewise_replay import validate_receipt as validate_piecewise_replay
from .rotating_external_benchmarks import validate_pack
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/retained_piecewise_descendant_campaign.json"
OUTPUT_PATH = "runs/math/retained-piecewise-descendant-campaign/live-runtime.json"
LIVE_CALL_JOURNAL_PATH = "work/retained-piecewise-descendant-campaign/live-call-attempts.jsonl"
SOURCE_PATH = "src/sigma_theory_compiler/retained_piecewise_descendant_campaign.py"
TRANSPORT_SOURCE_PATH = "src/sigma_theory_compiler/external_claude_transport.py"
TEST_PATH = "tests/test_retained_piecewise_descendant_campaign.py"
CONFIG_SCHEMA = "invariant-retained-piecewise-descendant-campaign-config-1.0"
SCHEMA_VERSION = "invariant-retained-piecewise-descendant-campaign-runtime-1.0"
PUBLIC_PAYLOAD_SCHEMA = "invariant-retained-piecewise-repair-public-payload-1.0"
CAMPAIGN_ROLES = (ClaudeRole.RECOMBINER, ClaudeRole.REPRESENTATION_INVENTOR)
CONTROL_SEED_NAMESPACE = "invariant.retained-piecewise-descendants.2026-08-24"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RetainedPiecewiseDescendantError(ValueError):
    """The live descendant campaign or its offline evidence failed closed."""


def _compact_fixed_provider_schema(
    schema: Mapping[str, Any], slot_names: Sequence[str]
) -> dict[str, Any]:
    try:
        hypotheses = schema["properties"]["hypotheses"]
        required = hypotheses["required"]
        properties = hypotheses["properties"]
    except (KeyError, TypeError) as error:
        raise ClaudeCreativityError("descendant fixed-hypothesis request schema changed") from error
    if (
        hypotheses.get("type") != "object"
        or tuple(required) != tuple(slot_names)
        or not isinstance(properties, Mapping)
        or set(properties) != set(slot_names)
        or any(properties[name] != properties[slot_names[0]] for name in slot_names[1:])
    ):
        raise ClaudeCreativityError("descendant fixed-hypothesis request allocation changed")
    compact_schema = json.loads(json.dumps(schema))
    compact_schema["properties"]["hypotheses"] = {
        "type": "array",
        "items": json.loads(json.dumps(properties[slot_names[0]])),
        "minItems": 1,
    }
    return compact_schema


class _FixedHypothesisProviderTransport:
    """Compact repeated fixed slots only for this campaign's provider wire contract."""

    def __init__(self, transport: Transport, hypothesis_slots: int) -> None:
        self.provider = ProviderCompatibleClaudeTransport(transport)
        self.slot_names = tuple(f"idea_{index}" for index in range(1, hypothesis_slots + 1))

    def evidence_for(self, response_id: str) -> Mapping[str, Any]:
        evidence = dict(self.provider.evidence_for(response_id))
        if evidence:
            evidence["fixed_hypothesis_adapter_used"] = True
            evidence["wire_contract_adapter_used"] = True
        return evidence

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        if method != "POST" or body is None:
            return self.provider(method, url, headers, body, timeout)
        try:
            request = json.loads(body)
            schema = request["output_config"]["format"]["schema"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ClaudeCreativityError(
                "descendant fixed-hypothesis request schema changed"
            ) from error
        request["output_config"]["format"]["schema"] = _compact_fixed_provider_schema(
            schema, self.slot_names
        )
        compact_body = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        status, response = self.provider(method, url, headers, compact_body, timeout)
        if status != 200 or response.get("type") != "message":
            return status, response
        content = response.get("content")
        if (
            not isinstance(content, list)
            or len(content) != 1
            or not isinstance(content[0], Mapping)
            or content[0].get("type") != "text"
            or not isinstance(content[0].get("text"), str)
        ):
            raise ClaudeCreativityError("descendant provider response content changed")
        try:
            output = json.loads(content[0]["text"])
        except json.JSONDecodeError as error:
            raise ClaudeCreativityError(
                "descendant provider structured text is not JSON"
            ) from error
        raw_hypotheses = output.get("hypotheses") if isinstance(output, dict) else None
        if (
            not isinstance(raw_hypotheses, list)
            or len(raw_hypotheses) != len(self.slot_names)
            or any(not isinstance(item, Mapping) for item in raw_hypotheses)
        ):
            raise ClaudeCreativityError(
                "descendant provider response changed fixed branch allocation"
            )
        adapted_output = dict(output)
        adapted_output["hypotheses"] = {
            name: dict(item) for name, item in zip(self.slot_names, raw_hypotheses, strict=True)
        }
        adapted_response = dict(response)
        adapted_response["content"] = [
            {
                "type": "text",
                "text": json.dumps(adapted_output, sort_keys=True, separators=(",", ":")),
            }
        ]
        return status, adapted_response


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _rooted_path(root: Path, value: str | Path) -> tuple[Path, str]:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise RetainedPiecewiseDescendantError("descendant campaign path escaped root") from error
    return resolved, relative


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetainedPiecewiseDescendantError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise RetainedPiecewiseDescendantError(f"{label} root is not an object")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RetainedPiecewiseDescendantError(f"{label} keys changed")


def load_config(root: Path, path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    resolved, _ = _rooted_path(root.resolve(), path)
    value = _load_json(resolved, "descendant campaign config")
    _strict(
        value,
        {
            "campaign_id",
            "claude",
            "hypotheses_per_call",
            "resource_budget",
            "roles",
            "schema_version",
            "sources",
            "task_bindings",
        },
        "descendant campaign config",
    )
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise RetainedPiecewiseDescendantError("descendant campaign config schema changed")
    if value.get("campaign_id") != "retained-piecewise-counterexample-repair-2026-08-24-004":
        raise RetainedPiecewiseDescendantError("descendant campaign identity changed")
    try:
        claude = ClaudeAPIConfig.from_mapping(value["claude"])
    except (ClaudeCreativityError, TypeError) as error:
        raise RetainedPiecewiseDescendantError("descendant Claude config changed") from error
    if (
        not claude.execution_enabled
        or claude.model != "claude-opus-4-6"
        or claude.credential_env_var != "ANTHROPIC_API_KEY"
        or claude.maximum_calls != 6
        or value.get("hypotheses_per_call") != 4
        or value.get("roles") != [role.value for role in CAMPAIGN_ROLES]
    ):
        raise RetainedPiecewiseDescendantError("live Claude repair policy changed")
    sources = value.get("sources")
    _strict(
        sources,
        {
            "fresh_generation_packet",
            "fresh_rotation_receipt",
            "fresh_target_packet",
            "retained_piecewise_replay",
            "source_core_receipt",
        },
        "descendant campaign sources",
    )
    for source_path in sources.values():
        if not isinstance(source_path, str):
            raise RetainedPiecewiseDescendantError("descendant source path is not a string")
        _rooted_path(root, source_path)
    budget = value.get("resource_budget")
    if budget != {
        "maximum_evaluation_operations": 1_000_000,
        "maximum_grammar_depth": 64,
        "maximum_verifier_invocations": 5,
    }:
        raise RetainedPiecewiseDescendantError("descendant resource budget changed")
    bindings = value.get("task_bindings")
    if not isinstance(bindings, list) or len(bindings) != 3:
        raise RetainedPiecewiseDescendantError("descendant task bindings changed")
    expected_sources = {"OEIS-A000330", "OEIS-A005132", "OEIS-A002858"}
    observed_sources = set()
    for binding in bindings:
        _strict(
            binding,
            {
                "evaluation_kind",
                "known_reference_formula",
                "parent_benchmark_ids",
                "source_id",
            },
            "descendant task binding",
        )
        source_id = binding.get("source_id")
        parents = binding.get("parent_benchmark_ids")
        if (
            source_id in observed_sources
            or source_id not in expected_sources
            or not isinstance(parents, list)
            or not parents
            or len(parents) != len(set(parents))
        ):
            raise RetainedPiecewiseDescendantError("descendant task binding is malformed")
        observed_sources.add(source_id)
        kind = binding.get("evaluation_kind")
        reference = binding.get("known_reference_formula")
        if (kind == "known_formula_control" and not isinstance(reference, str)) or (
            kind == "bounded_unknown" and reference is not None
        ):
            raise RetainedPiecewiseDescendantError("descendant evaluation kind changed")
    if observed_sources != expected_sources:
        raise RetainedPiecewiseDescendantError("descendant fresh task coverage changed")
    return value


def _sealed_core(path: Path) -> dict[str, Any]:
    value = _load_json(path, "source core receipt")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    archive = value.get("idea_lineage_archive")
    evidence = value.get("claude_runtime", {}).get("evidence")
    if (
        value.get("content_sha256") != canonical_sha256(body)
        or value.get("app_id") != "invariant.core-creative-discovery"
        or not isinstance(archive, Mapping)
        or not isinstance(evidence, Mapping)
        or value.get("claude_runtime", {}).get("authenticated_messages_api_working") is not True
    ):
        raise RetainedPiecewiseDescendantError("source core receipt is not reusable")
    validate_idea_archive(archive)
    return value


def _load_sources(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source_paths = config["sources"]
    generation = _load_json(root / source_paths["fresh_generation_packet"], "generation packet")
    targets = _load_json(root / source_paths["fresh_target_packet"], "target packet")
    rotation = _load_json(root / source_paths["fresh_rotation_receipt"], "rotation receipt")
    validate_pack(generation, targets, rotation, root)
    if generation.get("rotation_epoch") != "2026-08-24-003":
        raise RetainedPiecewiseDescendantError("fresh rotation epoch changed")
    replay = _load_json(root / source_paths["retained_piecewise_replay"], "piecewise replay")
    validate_piecewise_replay(replay, root)
    if replay.get("summary", {}).get("retained_piecewise_ideas") != 8:
        raise RetainedPiecewiseDescendantError("retained parent coverage changed")
    core = _sealed_core(root / source_paths["source_core_receipt"])
    return {
        "core": core,
        "generation": generation,
        "replay": replay,
        "rotation": rotation,
        "targets": targets,
    }


def _fresh_task(
    task: Mapping[str, Any],
    target: Mapping[str, Any],
    binding: Mapping[str, Any],
    rotation: Mapping[str, Any],
) -> tuple[Benchmark, SealedTarget]:
    training = tuple(
        Observation((Fraction(int(row["index"])),), Fraction(int(row["value"])))
        for row in task["training"]
    )
    holdout = tuple(
        Observation((Fraction(int(row["index"])),), Fraction(int(row["value"])))
        for row in target["holdout"]
    )
    kind = binding["evaluation_kind"]
    benchmark = Benchmark(
        task["task_id"],
        4 if kind == "known_formula_control" else 5,
        ExternalSource(
            "external.oeis-foundation",
            rotation["retrieved_utc"],
            "coordinator-sealed-rotation",
            "https://oeis.org",
        ),
        (Variable("index", (0, 0, 0), "integer_index"),),
        (0, 0, 0),
        "sequence_value",
        training,
        {},
        task["target_commitment"],
    )
    sealed = SealedTarget(
        task["task_id"],
        holdout,
        binding["known_reference_formula"],
        "known_formula" if kind == "known_formula_control" else "bounded_unknown",
        task["target_commitment"],
    )
    return benchmark, sealed


def _prior_counterexamples(root: Path, replay: Mapping[str, Any]) -> dict[str, Any]:
    public, benchmarks = load_public_benchmarks(root)
    targets = {item.benchmark_id: item for item in unseal_targets(root, public, benchmarks)}
    train_counts = {item.benchmark_id: len(item.observations) for item in benchmarks}
    counterexamples: dict[str, Any] = {}
    for row in replay["replays"]:
        target = targets[row["benchmark_id"]]
        predictions = row["execution"]["primary_predictions"]
        offset = train_counts[row["benchmark_id"]]
        witness = None
        for index, observation in enumerate(target.holdout_records):
            predicted = predictions[offset + index]
            expected = _fraction_text(observation.output)
            if predicted != expected:
                witness = {
                    "expected_output": expected,
                    "predicted_output": predicted,
                    "prior_row_inputs": {"x0": _fraction_text(observation.inputs[0])},
                }
                break
        counterexamples[row["lineage_id"]] = witness
    return counterexamples


def _parent_pool(
    root: Path,
    replay: Mapping[str, Any],
    core: Mapping[str, Any],
    affinity_benchmarks: Sequence[str],
) -> list[dict[str, Any]]:
    ideas = {item["lineage_id"]: item for item in core["idea_lineage_archive"]["ideas"]}
    counterexamples = _prior_counterexamples(root, replay)
    pool = []
    for ordinal, row in enumerate(
        sorted(replay["replays"], key=lambda item: item["lineage_id"]), 1
    ):
        idea = ideas.get(row["lineage_id"])
        if idea is None:
            raise RetainedPiecewiseDescendantError("retained parent lost source lineage")
        pool.append(
            {
                "known_analogues": list(idea.get("known_analogues", [])),
                "normalized_expression": row["execution"]["normalized_expression"],
                "origin_self_assessment": row["llm_self_assessed_origin"],
                "parent_ref": f"parent_{ordinal}",
                "prior_counterexample": counterexamples[row["lineage_id"]],
                "prior_freshness_note": "counterexample opened before the fresh rotation",
                "prior_train_loss": row["execution"]["train_loss"],
                "representation": "piecewise_relation",
                "source_domains": list(idea.get("source_idea_domains", [])),
                "task_affinity": row["benchmark_id"] in set(affinity_benchmarks),
            }
        )
    return pool


def _public_payload(
    root: Path,
    task: Mapping[str, Any],
    binding: Mapping[str, Any],
    replay: Mapping[str, Any],
    core: Mapping[str, Any],
) -> dict[str, Any]:
    parents = _parent_pool(root, replay, core, binding["parent_benchmark_ids"])
    return {
        "schema_version": PUBLIC_PAYLOAD_SCHEMA,
        "fresh_task": {
            "representation_family": task["representation_family"],
            "task_id": task["task_id"],
            "training": [
                {"inputs": {"x0": str(row["index"])}, "output": row["value"]}
                for row in task["training"]
            ],
            "variables": ["x0"],
        },
        "repair_policy": {
            "all_parent_branches_remain_active": True,
            "fresh_evaluation_values_visible_during_generation": False,
            "origin_labels_are_non_authoritative": True,
            "use_prior_counterexamples_for_repair": True,
        },
        "retained_parent_pool": parents,
    }


def _instruction() -> str:
    return (
        "Generate exactly four structurally distinct descendants of the retained parent pool for "
        "the anonymous fresh training rows. Use prior_counterexample fields to repair failed "
        "parents and cross-recombine mechanisms across task-affine and distant parents. In every "
        "synthesis_note name the parent_ref values used. Prefer compact falsifiable laws over "
        "per-index lookup tables. Include at least two executable piecewise_relation descendants "
        "and at least one representation-changing descendant when mathematically plausible. "
        "Self-label known_rewrite, cross_domain_synthesis, proposed_new_construction, or uncertain; "
        "labels never prune an idea and do not establish novelty. "
        + EXECUTABLE_PROPOSER_INSTRUCTION
    )


def _system_instruction() -> str:
    return (
        "You are the live creative repair component of the Invariant core app. Return only the "
        "required structured object. You see no fresh evaluation values or source identity. "
        "Propose and recombine freely, but do not claim verification, proof, or literature novelty."
    )


def _prompt_sha256(role: ClaudeRole, payload: Mapping[str, Any]) -> str:
    prompt = json.dumps(
        {
            "benchmark": payload,
            "candidate_summaries": [],
            "instruction": _instruction(),
            "role": role.value,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(prompt.encode()).hexdigest()


def _call_id(task_id: str, role: ClaudeRole) -> str:
    return "call." + canonical_sha256({"role": role.value, "task_id": task_id})[:24]


def _validate_activation(value: Mapping[str, Any]) -> None:
    _strict(
        value,
        {
            "credential_env_var",
            "credential_persisted",
            "credential_value_recorded",
            "injected_into_process",
            "source_kind",
            "source_locator_sha256",
        },
        "credential activation",
    )
    if (
        value.get("credential_env_var") != "ANTHROPIC_API_KEY"
        or value.get("credential_persisted") is not False
        or value.get("credential_value_recorded") is not False
        or value.get("source_kind")
        not in {
            "process_environment",
            "explicit_env_file",
            "project_env_file",
            "user_invariant_env_file",
        }
        or _SHA256.fullmatch(str(value.get("source_locator_sha256"))) is None
        or not isinstance(value.get("injected_into_process"), bool)
    ):
        raise RetainedPiecewiseDescendantError("credential activation evidence changed")


def _validate_call(
    call: Mapping[str, Any],
    *,
    expected_role: ClaudeRole,
    expected_task_id: str,
    expected_payload: Mapping[str, Any],
    expected_parent_ids: Sequence[str],
    expected_affinity_ids: Sequence[str],
    config: Mapping[str, Any],
) -> list[ClaudeHypothesis]:
    _strict(
        call,
        {
            "call_id",
            "evidence",
            "output",
            "parent_lineage_ids",
            "provider_transport",
            "public_payload",
            "role",
            "status",
            "task_affinity_parent_lineage_ids",
            "task_id",
        },
        "Claude descendant call",
    )
    if (
        call.get("call_id") != _call_id(expected_task_id, expected_role)
        or call.get("role") != expected_role.value
        or call.get("task_id") != expected_task_id
        or call.get("status") != ClaudeCallStatus.COMPLETED.value
        or call.get("public_payload") != expected_payload
        or call.get("parent_lineage_ids") != list(expected_parent_ids)
        or call.get("task_affinity_parent_lineage_ids") != list(expected_affinity_ids)
    ):
        raise RetainedPiecewiseDescendantError("Claude descendant call identity changed")
    evidence = call.get("evidence")
    output = call.get("output")
    provider = call.get("provider_transport")
    if (
        not isinstance(evidence, Mapping)
        or not isinstance(output, Mapping)
        or not isinstance(provider, Mapping)
    ):
        raise RetainedPiecewiseDescendantError("Claude descendant call evidence is malformed")
    expected_schema = _structured_output_schema(
        expected_role,
        expected_task_id,
        bind_role_collections=True,
        hypothesis_slots=config["hypotheses_per_call"],
    )
    slot_names = tuple(f"idea_{index}" for index in range(1, config["hypotheses_per_call"] + 1))
    expected_provider_schema = _compact_fixed_provider_schema(expected_schema, slot_names)
    if (
        evidence.get("credential_persisted") is not False
        or evidence.get("model") != config["claude"]["model"]
        or evidence.get("prompt_sha256") != _prompt_sha256(expected_role, expected_payload)
        or evidence.get("request_schema_sha256") != canonical_sha256(expected_schema)
        or evidence.get("header_names") != ["anthropic-version", "content-type", "x-api-key"]
        or evidence.get("model_evidence", {}).get("structured_outputs_supported") is not True
        or not isinstance(evidence.get("api_response_id"), str)
        or not evidence["api_response_id"]
        or provider.get("provider_header_names")
        != ["anthropic-version", "content-type", "user-agent", "x-api-key"]
        or provider.get("fixed_hypothesis_adapter_used") is not True
        or provider.get("wire_contract_adapter_used") is not True
        or provider.get("provider_prompt_sha256") != evidence.get("prompt_sha256")
        or provider.get("provider_request_schema_sha256")
        != canonical_sha256(expected_provider_schema)
    ):
        raise RetainedPiecewiseDescendantError("Claude runtime-health evidence changed")
    _strict(
        output,
        {"benchmark_id", "hypotheses", "quarantine", "role", "schema_version", "steering_actions"},
        "Claude descendant output",
    )
    if (
        output.get("benchmark_id") != expected_task_id
        or output.get("role") != expected_role.value
        or output.get("schema_version") != "invariant-claude-creativity-output-2.0"
        or output.get("steering_actions") != []
        or output.get("quarantine", {}).get("rejected_hypotheses") != 0
        or evidence.get("output_sha256") != canonical_sha256(output)
    ):
        raise RetainedPiecewiseDescendantError("Claude descendant output binding changed")
    raw = output.get("hypotheses")
    if not isinstance(raw, list) or len(raw) != config["hypotheses_per_call"]:
        raise RetainedPiecewiseDescendantError("Claude descendant branch allocation changed")
    provider_raw_output = {
        "benchmark_id": output["benchmark_id"],
        "hypotheses": raw,
        "role": output["role"],
        "schema_version": output["schema_version"],
        "steering_actions": {},
    }
    if provider.get("provider_raw_output_sha256") != canonical_sha256(provider_raw_output):
        raise RetainedPiecewiseDescendantError("Claude provider output binding changed")
    try:
        return [ClaudeHypothesis.from_mapping(item) for item in raw]
    except (ClaudeCreativityError, TypeError) as error:
        raise RetainedPiecewiseDescendantError(
            "Claude descendant hypothesis is malformed"
        ) from error


def _predictions(values: Sequence[Fraction | None]) -> list[str | None]:
    return [None if value is None else _fraction_text(value) for value in values]


def _referenced_parent_lineage(
    hypothesis: ClaudeHypothesis, call: Mapping[str, Any]
) -> tuple[list[str], str]:
    """Resolve public parent aliases without inventing lineage the model did not state."""

    available = call["parent_lineage_ids"]
    raw_refs = re.findall(r"\bparent_([0-9]+)\b", hypothesis.synthesis_note)
    ordinals = sorted({int(value) for value in raw_refs})
    if not ordinals:
        return [], "MISSING_EXPLICIT_PARENT_REF"
    if any(ordinal < 1 or ordinal > len(available) for ordinal in ordinals):
        return [], "INVALID_PARENT_REF_RETAINED"
    return [available[ordinal - 1] for ordinal in ordinals], "EXPLICIT_PARENT_REFS_RESOLVED"


def _descendant_row(
    hypothesis: ClaudeHypothesis,
    *,
    call: Mapping[str, Any],
    ordinal: int,
    benchmark: Benchmark,
    target: SealedTarget,
    budget: Mapping[str, int],
) -> dict[str, Any]:
    hypothesis_dict = hypothesis.to_dict()
    descendant_id = (
        "descendant."
        + canonical_sha256(
            {"call_id": call["call_id"], "hypothesis": hypothesis_dict, "ordinal": ordinal}
        )[:24]
    )
    candidate, admission = _claude_candidate(benchmark, hypothesis)
    referenced_parents, lineage_status = _referenced_parent_lineage(hypothesis, call)
    row: dict[str, Any] = {
        "admission": admission,
        "available_parent_lineage_ids": list(call["parent_lineage_ids"]),
        "available_task_affinity_parent_lineage_ids": list(
            call["task_affinity_parent_lineage_ids"]
        ),
        "descendant_id": descendant_id,
        "generation_call_id": call["call_id"],
        "hypothesis": hypothesis_dict,
        "lineage_parse_status": lineage_status,
        "llm_self_assessed_origin": hypothesis.llm_origin_assessment,
        "parent_lineage_ids": referenced_parents,
        "retention_status": "RETAINED_ACTIVE",
        "task_id": benchmark.benchmark_id,
        "target_kind": target.target_kind,
    }
    if candidate is None:
        return row
    rows = (*benchmark.observations, *target.holdout_records)
    primary = predict(candidate, benchmark, rows)
    independent = independently_predict(candidate, benchmark, rows)
    seed = int(
        hashlib.sha256(f"{CONTROL_SEED_NAMESPACE}:{descendant_id}".encode()).hexdigest()[:16], 16
    )
    control = random_controls(benchmark, {descendant_id: (candidate,)}, seed)[descendant_id][0]
    candidate_profile = _candidate_resource_profile(candidate, benchmark, target, budget)
    control_profile = _candidate_resource_profile(control, benchmark, target, budget)
    train_count = len(benchmark.observations)
    proof_search = _proof_plan_search(candidate, target, budget["maximum_verifier_invocations"])
    row["execution"] = {
        "behavior": _behavior(candidate, benchmark, rows),
        "candidate_id": candidate.candidate_id,
        "candidate_resource_profile": candidate_profile,
        "control_behavior": _behavior(control, benchmark, rows),
        "control_candidate_id": control.candidate_id,
        "control_resource_profile": control_profile,
        "fresh_holdout_loss": _fraction_text(_loss(primary[train_count:], target.holdout_records)),
        "fresh_train_loss": _fraction_text(_loss(primary[:train_count], benchmark.observations)),
        "independent_predictions": _predictions(independent),
        "normalized_expression": candidate.expression,
        "primary_independent_exact_agreement": primary == independent,
        "primary_predictions": _predictions(primary),
        "proof_plan_search": proof_search,
        "resource_profile_exact_match": candidate_profile == control_profile,
        "undefined_rows": sum(value is None for value in primary),
    }
    return row


def _expected_call_contexts(
    root: Path, config: Mapping[str, Any], sources: Mapping[str, Any]
) -> list[dict[str, Any]]:
    generation = sources["generation"]
    targets = sources["targets"]
    replay = sources["replay"]
    core = sources["core"]
    task_by_source = {
        target["source_id"]: (
            next(task for task in generation["tasks"] if task["task_id"] == target["task_id"]),
            target,
        )
        for target in targets["targets"]
    }
    replay_by_benchmark: dict[str, list[str]] = {}
    all_parent_ids = sorted(item["lineage_id"] for item in replay["replays"])
    for item in replay["replays"]:
        replay_by_benchmark.setdefault(item["benchmark_id"], []).append(item["lineage_id"])
    contexts = []
    for binding in config["task_bindings"]:
        try:
            task, target = task_by_source[binding["source_id"]]
        except KeyError as error:
            raise RetainedPiecewiseDescendantError("fresh rotation lost a bound task") from error
        affinity = sorted(
            lineage_id
            for benchmark_id in binding["parent_benchmark_ids"]
            for lineage_id in replay_by_benchmark.get(benchmark_id, [])
        )
        payload = _public_payload(root, task, binding, replay, core)
        benchmark, sealed = _fresh_task(task, target, binding, sources["rotation"])
        for role in CAMPAIGN_ROLES:
            contexts.append(
                {
                    "affinity": affinity,
                    "all_parents": all_parent_ids,
                    "benchmark": benchmark,
                    "binding": binding,
                    "payload": payload,
                    "role": role,
                    "target": sealed,
                    "task": task,
                }
            )
    return contexts


def _build_from_calls(
    root: Path,
    config: Mapping[str, Any],
    sources: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
    credential_activation: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_activation(credential_activation)
    contexts = _expected_call_contexts(root, config, sources)
    if len(calls) != len(contexts) or len(contexts) != 6:
        raise RetainedPiecewiseDescendantError("descendant call coverage changed")
    descendants = []
    normalized_calls = []
    response_ids = []
    usage = Counter()
    for call, context in zip(calls, contexts, strict=True):
        normalized_call = dict(call)
        provider_transport = dict(call.get("provider_transport", {}))
        provider_transport["fixed_hypothesis_adapter_used"] = True
        provider_transport["wire_contract_adapter_used"] = True
        normalized_call["provider_transport"] = provider_transport
        hypotheses = _validate_call(
            normalized_call,
            expected_role=context["role"],
            expected_task_id=context["task"]["task_id"],
            expected_payload=context["payload"],
            expected_parent_ids=context["all_parents"],
            expected_affinity_ids=context["affinity"],
            config=config,
        )
        normalized_calls.append(normalized_call)
        response_ids.append(normalized_call["evidence"]["api_response_id"])
        for key in ("input_tokens", "output_tokens"):
            amount = normalized_call["evidence"].get("usage", {}).get(key)
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise RetainedPiecewiseDescendantError("Claude usage evidence changed")
            usage[key] += amount
        descendants.extend(
            _descendant_row(
                hypothesis,
                call=normalized_call,
                ordinal=ordinal,
                benchmark=context["benchmark"],
                target=context["target"],
                budget=config["resource_budget"],
            )
            for ordinal, hypothesis in enumerate(hypotheses, 1)
        )
    if len(set(response_ids)) != len(response_ids):
        raise RetainedPiecewiseDescendantError("Claude response IDs are not distinct")
    admitted = [item for item in descendants if "execution" in item]
    parent_behaviors = {
        item["execution"]["behavior"]["behavior_sha256"] for item in sources["replay"]["replays"]
    }
    parent_mechanisms = {
        item["execution"]["behavior"]["proof_mechanism_sha256"]
        for item in sources["replay"]["replays"]
    }
    descendant_behaviors = {item["execution"]["behavior"]["behavior_sha256"] for item in admitted}
    descendant_mechanisms = {
        item["execution"]["behavior"]["proof_mechanism_sha256"] for item in admitted
    }
    source_paths = config["sources"]
    parent_prediction_archive = [
        {
            "behavior": item["execution"]["behavior"],
            "benchmark_id": item["benchmark_id"],
            "independent_predictions": item["execution"]["independent_predictions"],
            "lineage_id": item["lineage_id"],
            "primary_predictions": item["execution"]["primary_predictions"],
        }
        for item in sources["replay"]["replays"]
    ]
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "config": {
                "path": CONFIG_PATH,
                "sha256": _normalized_file_sha256(root / CONFIG_PATH),
            },
            "fresh_generation_packet": {
                "content_sha256": sources["generation"]["content_sha256"],
                "path": source_paths["fresh_generation_packet"],
            },
            "fresh_rotation_receipt": {
                "content_sha256": sources["rotation"]["content_sha256"],
                "path": source_paths["fresh_rotation_receipt"],
            },
            "fresh_target_packet": {
                "content_sha256": sources["targets"]["content_sha256"],
                "path": source_paths["fresh_target_packet"],
            },
            "retained_piecewise_replay": {
                "content_sha256": sources["replay"]["content_sha256"],
                "path": source_paths["retained_piecewise_replay"],
            },
            "source_core_lineage": {
                "idea_lineage_archive_sha256": sources["core"]["idea_lineage_archive"][
                    "content_sha256"
                ],
                "live_evidence_sha256": sources["core"]["claude_runtime"]["evidence"][
                    "content_sha256"
                ],
                "path": source_paths["source_core_receipt"],
            },
            "provider_transport": {
                "path": TRANSPORT_SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / TRANSPORT_SOURCE_PATH),
            },
            "source": {"path": SOURCE_PATH, "sha256": _normalized_file_sha256(root / SOURCE_PATH)},
            "tests": {"path": TEST_PATH, "sha256": _normalized_file_sha256(root / TEST_PATH)},
        },
        "credential_activation": dict(credential_activation),
        "claude_runtime": {
            "authenticated_messages_api_working": True,
            "completed_calls": len(calls),
            "distinct_response_ids": len(set(response_ids)),
            "model": config["claude"]["model"],
            "roles_completed": sorted({call["role"] for call in calls}),
            "status": "PASS_LIVE_CORE_DESCENDANT_PARTICIPATION",
            "structured_outputs_supported": True,
            "usage": {
                "calls": len(calls),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            },
        },
        "claude_calls": normalized_calls,
        "descendants": descendants,
        "parent_prediction_archive": parent_prediction_archive,
        "novelty_axes": {
            "behavior_novelty_is_literature_novelty": False,
            "distinct_descendant_behavior_signatures": len(descendant_behaviors),
            "distinct_descendant_proof_mechanism_signatures": len(descendant_mechanisms),
            "new_behavior_signatures_vs_parents": len(descendant_behaviors - parent_behaviors),
            "new_proof_mechanism_signatures_vs_parents": len(
                descendant_mechanisms - parent_mechanisms
            ),
            "proof_mechanism_novelty_is_literature_novelty": False,
            "separate_axes": True,
        },
        "summary": {
            "admitted_executable_descendants": len(admitted),
            "descendant_ideas_retained": len(descendants),
            "exact_primary_independent_agreements": sum(
                item["execution"]["primary_independent_exact_agreement"] for item in admitted
            ),
            "fresh_tasks": len(config["task_bindings"]),
            "llm_self_assessed_origin_counts": dict(
                sorted(Counter(item["llm_self_assessed_origin"] for item in descendants).items())
            ),
            "nonexecutable_descendants_retained": len(descendants) - len(admitted),
            "descendants_with_explicit_parent_lineage": sum(
                item["lineage_parse_status"] == "EXPLICIT_PARENT_REFS_RESOLVED"
                for item in descendants
            ),
            "parent_branches_exposed": len(
                {parent for item in descendants for parent in item["available_parent_lineage_ids"]}
            ),
            "parent_branches_preserved": len(parent_prediction_archive),
            "parent_branches_used_by_explicit_lineage": len(
                {parent for item in descendants for parent in item["parent_lineage_ids"]}
            ),
            "representation_counts": dict(
                sorted(
                    Counter(item["hypothesis"]["representation"] for item in descendants).items()
                )
            ),
            "resource_matched_controls": sum(
                item["execution"]["resource_profile_exact_match"] for item in admitted
            ),
            "status": "PASS_LIVE_RETAINED_PIECEWISE_DESCENDANT_CAMPAIGN",
            "zero_fresh_holdout_bounded_unknown": sum(
                item["target_kind"] == "bounded_unknown"
                and item["execution"]["fresh_holdout_loss"] == "0"
                for item in admitted
            ),
            "zero_fresh_holdout_known_control": sum(
                item["target_kind"] == "known_formula"
                and item["execution"]["fresh_holdout_loss"] == "0"
                for item in admitted
            ),
            "zero_fresh_train_descendants": sum(
                item["execution"]["fresh_train_loss"] == "0" for item in admitted
            ),
        },
        "claim_boundary": {
            "executor_admission_establishes_correctness": False,
            "fresh_fit_establishes_general_formula": False,
            "llm_origin_assessment_establishes_novelty": False,
            "model_participation_establishes_creativity_superiority": False,
            "unsigned_rotation_counts_as_level5": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def run_live(
    root: Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    transport: Transport = urllib_transport,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
    attempt_journal_path: Path | None = None,
) -> dict[str, Any]:
    """Execute six authenticated creative calls and return credential-free evidence."""

    root = root.resolve()
    config = load_config(root, config_path)
    sources = _load_sources(root, config)
    contexts = _expected_call_contexts(root, config, sources)
    adapter = _FixedHypothesisProviderTransport(transport, config["hypotheses_per_call"])
    client = ClaudeCreativityClient(ClaudeAPIConfig.from_mapping(config["claude"]), adapter)
    calls = []
    attempt_id = uuid.uuid4().hex if attempt_journal_path is not None else None

    def journal(kind: str, payload: Mapping[str, Any]) -> None:
        if attempt_journal_path is None:
            return
        attempt_journal_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "attempt_id": attempt_id,
            "campaign_id": config["campaign_id"],
            "kind": kind,
            **payload,
        }
        with attempt_journal_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    journal("attempt_started", {"planned_calls": len(contexts)})
    with activated_credential(
        project_root=root,
        env_var=config["claude"]["credential_env_var"],
        environment=environment,
        home=home,
    ) as activation:
        for context in contexts:
            result = client.run(
                context["role"],
                context["task"]["task_id"],
                context["payload"],
                instruction_override=_instruction(),
                system_override=_system_instruction(),
                hypothesis_slots=config["hypotheses_per_call"],
            )
            if result.status is not ClaudeCallStatus.COMPLETED or result.output is None:
                raise RetainedPiecewiseDescendantError(
                    "live Claude descendant call did not complete"
                )
            record = result.to_dict()
            response_id = record["evidence"]["api_response_id"]
            call_record = {
                "call_id": _call_id(context["task"]["task_id"], context["role"]),
                "evidence": record["evidence"],
                "output": record["output"],
                "parent_lineage_ids": list(context["all_parents"]),
                "provider_transport": dict(adapter.evidence_for(response_id)),
                "public_payload": context["payload"],
                "role": context["role"].value,
                "status": record["status"],
                "task_affinity_parent_lineage_ids": list(context["affinity"]),
                "task_id": context["task"]["task_id"],
            }
            calls.append(call_record)
            journal(
                "claude_call_completed",
                {"call_index": len(calls) - 1, "call": call_record},
            )
        activation_evidence = activation.to_evidence()
    receipt = _build_from_calls(root, config, sources, calls, activation_evidence)
    journal(
        "attempt_completed",
        {"campaign_content_sha256": receipt["content_sha256"], "completed_calls": len(calls)},
    )
    return receipt


def rebind_receipt(root: Path, previous: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild deterministic execution and source seals without another provider call."""

    root = root.resolve()
    config = load_config(root)
    sources = _load_sources(root, config)
    calls = previous.get("claude_calls")
    activation = previous.get("credential_activation")
    if not isinstance(calls, list) or not isinstance(activation, Mapping):
        raise RetainedPiecewiseDescendantError("previous descendant receipt lacks live evidence")
    return _build_from_calls(root, config, sources, calls, activation)


def validate_receipt(value: Mapping[str, Any], root: Path) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("content_sha256") != canonical_sha256(body)
        or value.get("summary", {}).get("status")
        != "PASS_LIVE_RETAINED_PIECEWISE_DESCENDANT_CAMPAIGN"
        or value.get("claude_runtime", {}).get("authenticated_messages_api_working") is not True
        or value.get("claude_runtime", {}).get("completed_calls") != 6
        or value.get("summary", {}).get("parent_branches_exposed") != 8
        or value.get("summary", {}).get("parent_branches_preserved") != 8
        or value.get("claim_boundary")
        != {
            "executor_admission_establishes_correctness": False,
            "fresh_fit_establishes_general_formula": False,
            "llm_origin_assessment_establishes_novelty": False,
            "model_participation_establishes_creativity_superiority": False,
            "unsigned_rotation_counts_as_level5": False,
        }
    ):
        raise RetainedPiecewiseDescendantError("descendant campaign policy or seal changed")
    expected = rebind_receipt(root, value)
    if dict(value) != expected:
        raise RetainedPiecewiseDescendantError("descendant campaign does not replay exactly")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute the authenticated descendant campaign")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    run.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    rebind = subparsers.add_parser(
        "rebind", help="rebind stored live calls without provider access"
    )
    rebind.add_argument("--root", type=Path, default=Path.cwd())
    rebind.add_argument("--previous", type=Path, default=Path(OUTPUT_PATH))
    rebind.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate", help="validate a stored descendant receipt")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "run":
        receipt = run_live(
            root,
            config_path=args.config,
            attempt_journal_path=root / LIVE_CALL_JOURNAL_PATH,
        )
    else:
        source = args.previous if args.command == "rebind" else args.receipt
        source_path, _ = _rooted_path(root, source)
        previous = _load_json(source_path, "stored descendant receipt")
        if args.command == "validate":
            validate_receipt(previous, root)
            receipt = previous
        else:
            receipt = rebind_receipt(root, previous)
    if args.command != "validate":
        output_path, _ = _rooted_path(root, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {"claude_runtime": receipt["claude_runtime"], "summary": receipt["summary"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "OUTPUT_PATH",
    "RetainedPiecewiseDescendantError",
    "load_config",
    "main",
    "rebind_receipt",
    "run_live",
    "validate_receipt",
]
