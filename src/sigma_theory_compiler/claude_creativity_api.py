"""Fail-closed Claude Messages API boundary for mathematical creativity campaigns.

Claude is deliberately limited to proposal and critique.  The persisted trace binds the
public prompt, model, structured response, and token usage, but never the credential.  A
Claude response is evidence that a model participated; it is never mathematical verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .sigma_core import canonical_sha256

ANTHROPIC_VERSION = "2023-06-01"
MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"
MODELS_ENDPOINT = "https://api.anthropic.com/v1/models"
CLAUDE_OUTPUT_SCHEMA_VERSION = "invariant-claude-creativity-output-2.0"
CLAUDE_ORIGIN_ASSESSMENTS = {
    "cross_domain_synthesis",
    "known_rewrite",
    "proposed_new_construction",
    "uncertain",
}
CLAUDE_REPRESENTATIONS = {
    "finite_product",
    "finite_sum",
    "generating_function",
    "invariant_relation",
    "linear_recurrence",
    "modular_relation",
    "other_typed_relation",
    "proof_plan",
    "sympy_expression",
    "tensor_identity",
    "transform_relation",
    "variational_principle",
}
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")


class ClaudeCreativityError(ValueError):
    """The Claude boundary, budget, or structured output failed closed."""


class ClaudeRole(str, Enum):
    PROPOSER = "proposer"
    CRITIC = "critic"
    ANALOGUE_SCOUT = "analogue_scout"
    DATASET_EXPLAINER = "dataset_explainer"
    PROOF_STRATEGIST = "proof_strategist"
    RECOMBINER = "recombiner"
    REPRESENTATION_INVENTOR = "representation_inventor"


class ClaudeCallStatus(str, Enum):
    COMPLETED = "completed"
    BLOCKED_DISABLED = "blocked_disabled"
    BLOCKED_MISSING_CREDENTIAL = "blocked_missing_credential"


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ClaudeCreativityError(f"{label} keys changed")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ClaudeCreativityError(f"{label} is not a portable identifier")
    return value


def _text(value: Any, label: str, *, maximum_bytes: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value.encode("utf-8")) > maximum_bytes
    ):
        raise ClaudeCreativityError(f"{label} is empty, unstripped, or oversized")
    return value


def _text_array(value: Any, label: str, *, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ClaudeCreativityError(f"{label} is not a bounded JSON array")
    result = tuple(_text(item, label, maximum_bytes=512) for item in value)
    if len(set(result)) != len(result):
        raise ClaudeCreativityError(f"{label} contains duplicates")
    return result


def _contains_forbidden_key(value: Any) -> bool:
    forbidden = {
        "answer",
        "holdout",
        "reference_formula",
        "source_answer",
        "target",
        "target_reveal",
    }
    if isinstance(value, Mapping):
        return any(str(key).lower() in forbidden or _contains_forbidden_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class ClaudeAPIConfig:
    model: str
    credential_env_var: str = "ANTHROPIC_API_KEY"
    execution_enabled: bool = False
    maximum_calls: int = 8
    maximum_total_tokens: int = 64_000
    maximum_output_tokens: int = 4_096
    timeout_seconds: int = 90
    effort: str = "high"

    def __post_init__(self) -> None:
        _identifier(self.model, "Claude model")
        if _ENVIRONMENT_NAME.fullmatch(self.credential_env_var) is None:
            raise ClaudeCreativityError("Claude credential reference is invalid")
        if not isinstance(self.execution_enabled, bool):
            raise ClaudeCreativityError("Claude execution flag must be boolean")
        if not 1 <= self.maximum_calls <= 64:
            raise ClaudeCreativityError("Claude call cap must be in [1, 64]")
        if not 1_024 <= self.maximum_total_tokens <= 2_000_000:
            raise ClaudeCreativityError("Claude total-token cap is outside policy")
        if not 256 <= self.maximum_output_tokens <= 32_768:
            raise ClaudeCreativityError("Claude output-token cap is outside policy")
        if not 1 <= self.timeout_seconds <= 300:
            raise ClaudeCreativityError("Claude timeout is outside policy")
        if self.effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ClaudeCreativityError("Claude effort is unsupported")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClaudeAPIConfig:
        keys = {
            "credential_env_var",
            "effort",
            "execution_enabled",
            "maximum_calls",
            "maximum_output_tokens",
            "maximum_total_tokens",
            "model",
            "timeout_seconds",
        }
        _strict_keys(value, keys, "Claude API config")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "credential_env_var": self.credential_env_var,
            "effort": self.effort,
            "execution_enabled": self.execution_enabled,
            "maximum_calls": self.maximum_calls,
            "maximum_output_tokens": self.maximum_output_tokens,
            "maximum_total_tokens": self.maximum_total_tokens,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class ClaudeBudget:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, input_tokens: int, output_tokens: int) -> ClaudeBudget:
        if min(input_tokens, output_tokens) < 0:
            raise ClaudeCreativityError("Claude usage contains a negative token count")
        return ClaudeBudget(
            self.calls + 1,
            self.input_tokens + input_tokens,
            self.output_tokens + output_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ClaudeHypothesis:
    hypothesis_id: str
    family: str
    representation: str
    expression: str
    invariants: tuple[str, ...]
    proof_plan: tuple[str, ...]
    falsifiers: tuple[str, ...]
    rationale: str
    llm_origin_assessment: str
    known_analogues: tuple[str, ...]
    source_idea_domains: tuple[str, ...]
    synthesis_note: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClaudeHypothesis:
        _strict_keys(
            value,
            {
                "expression",
                "falsifiers",
                "family",
                "hypothesis_id",
                "invariants",
                "known_analogues",
                "llm_origin_assessment",
                "proof_plan",
                "rationale",
                "representation",
                "source_idea_domains",
                "synthesis_note",
            },
            "Claude hypothesis",
        )
        representation = _identifier(value["representation"], "Claude representation")
        if representation not in CLAUDE_REPRESENTATIONS:
            raise ClaudeCreativityError("Claude representation is not admitted")
        origin_assessment = _identifier(
            value["llm_origin_assessment"], "Claude origin assessment"
        )
        if origin_assessment not in CLAUDE_ORIGIN_ASSESSMENTS:
            raise ClaudeCreativityError("Claude origin assessment is not admitted")
        expression = _text(value["expression"], "Claude expression", maximum_bytes=512)
        return cls(
            _identifier(value["hypothesis_id"], "Claude hypothesis_id"),
            _identifier(value["family"], "Claude family"),
            representation,
            expression,
            _text_array(value["invariants"], "Claude invariants", maximum=16),
            _text_array(value["proof_plan"], "Claude proof plan", maximum=16),
            _text_array(value["falsifiers"], "Claude falsifiers", maximum=16),
            _text(value["rationale"], "Claude rationale", maximum_bytes=2048),
            origin_assessment,
            _text_array(value["known_analogues"], "Claude known analogues", maximum=16),
            _text_array(
                value["source_idea_domains"], "Claude source idea domains", maximum=16
            ),
            _text(value["synthesis_note"], "Claude synthesis note", maximum_bytes=2048),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "falsifiers": list(self.falsifiers),
            "family": self.family,
            "hypothesis_id": self.hypothesis_id,
            "invariants": list(self.invariants),
            "known_analogues": list(self.known_analogues),
            "llm_origin_assessment": self.llm_origin_assessment,
            "proof_plan": list(self.proof_plan),
            "rationale": self.rationale,
            "representation": self.representation,
            "source_idea_domains": list(self.source_idea_domains),
            "synthesis_note": self.synthesis_note,
        }


@dataclass(frozen=True, slots=True)
class ClaudeSteeringAction:
    candidate_id: str
    verdict: str
    blocker_kind: str
    distance_numerator: int
    distance_denominator: int
    repair: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClaudeSteeringAction:
        _strict_keys(
            value,
            {
                "blocker_kind",
                "candidate_id",
                "distance_denominator",
                "distance_numerator",
                "repair",
                "verdict",
            },
            "Claude steering action",
        )
        verdict = value["verdict"]
        if verdict not in {"reject", "repair", "retain"}:
            raise ClaudeCreativityError("Claude steering verdict is invalid")
        numerator = value["distance_numerator"]
        denominator = value["distance_denominator"]
        if (
            isinstance(numerator, bool)
            or isinstance(denominator, bool)
            or not isinstance(numerator, int)
            or not isinstance(denominator, int)
            or numerator < 0
            or denominator <= 0
            or numerator > denominator
        ):
            raise ClaudeCreativityError("Claude steering distance is not in [0, 1]")
        repair = value["repair"]
        if (
            not isinstance(repair, str)
            or repair != repair.strip()
            or len(repair.encode("utf-8")) > 2048
            or (verdict == "repair" and not repair)
        ):
            raise ClaudeCreativityError("Claude steering repair text is invalid")
        return cls(
            _identifier(value["candidate_id"], "Claude candidate_id"),
            verdict,
            _identifier(value["blocker_kind"], "Claude blocker kind"),
            numerator,
            denominator,
            repair,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocker_kind": self.blocker_kind,
            "candidate_id": self.candidate_id,
            "distance_denominator": self.distance_denominator,
            "distance_numerator": self.distance_numerator,
            "repair": self.repair,
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class ClaudeStructuredOutput:
    role: ClaudeRole
    benchmark_id: str
    hypotheses: tuple[ClaudeHypothesis, ...]
    steering_actions: tuple[ClaudeSteeringAction, ...]
    rejected_hypotheses: int = 0
    rejected_steering_actions: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClaudeStructuredOutput:
        _strict_keys(
            value,
            {"benchmark_id", "hypotheses", "role", "schema_version", "steering_actions"},
            "Claude structured output",
        )
        if value["schema_version"] != CLAUDE_OUTPUT_SCHEMA_VERSION:
            raise ClaudeCreativityError("Claude output schema_version changed")
        role = ClaudeRole(value["role"])
        hypotheses_raw = value["hypotheses"]
        steering_raw = value["steering_actions"]
        if not isinstance(hypotheses_raw, list) or not isinstance(steering_raw, list):
            raise ClaudeCreativityError("Claude output collections must be JSON arrays")
        hypotheses_list = []
        rejected_hypotheses = 0
        for item in hypotheses_raw:
            try:
                hypotheses_list.append(ClaudeHypothesis.from_mapping(item))
            except ClaudeCreativityError:
                rejected_hypotheses += 1
        actions_list = []
        rejected_actions = 0
        for item in steering_raw:
            try:
                actions_list.append(ClaudeSteeringAction.from_mapping(item))
            except ClaudeCreativityError:
                rejected_actions += 1
        hypotheses = tuple(hypotheses_list)
        actions = tuple(actions_list)
        if len(hypotheses) > 16 or len(actions) > 64:
            raise ClaudeCreativityError("Claude output exceeds the bounded campaign size")
        return cls(
            role,
            _identifier(value["benchmark_id"], "Claude benchmark_id"),
            hypotheses,
            actions,
            rejected_hypotheses,
            rejected_actions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "role": self.role.value,
            "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
            "steering_actions": [item.to_dict() for item in self.steering_actions],
            "quarantine": {
                "rejected_hypotheses": self.rejected_hypotheses,
                "rejected_steering_actions": self.rejected_steering_actions,
            },
        }


@dataclass(frozen=True, slots=True)
class ClaudeCallResult:
    status: ClaudeCallStatus
    role: ClaudeRole
    benchmark_id: str
    output: ClaudeStructuredOutput | None
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "evidence": dict(self.evidence),
            "output": None if self.output is None else self.output.to_dict(),
            "role": self.role.value,
            "status": self.status.value,
        }


Transport = Callable[
    [str, str, Mapping[str, str], bytes | None, float], tuple[int, Mapping[str, Any]]
]


def urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, Mapping[str, Any]]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ClaudeCreativityError("Claude response exceeded the hard byte cap")
            parsed = json.loads(payload)
            if not isinstance(parsed, Mapping):
                raise ClaudeCreativityError("Claude response envelope is not an object")
            return int(response.status), parsed
    except urllib.error.HTTPError as error:
        payload = error.read(8_192)
        detail = "unparseable_error_envelope"
        try:
            parsed_error = json.loads(payload)
            error_body = parsed_error.get("error", {})
            if isinstance(error_body, Mapping):
                error_type = str(error_body.get("type", "unknown_error"))[:128]
                error_message = str(error_body.get("message", "no_message"))[:512]
                detail = f"{error_type}: {error_message}"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        raise ClaudeCreativityError(f"Claude API returned HTTP {error.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ClaudeCreativityError(f"Claude API transport failed: {type(error).__name__}") from None


def _structured_output_schema(role: ClaudeRole, benchmark_id: str) -> dict[str, Any]:
    hypothesis = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
            "falsifiers": {"type": "array", "items": {"type": "string"}},
            "family": {"type": "string"},
            "hypothesis_id": {"type": "string"},
            "invariants": {"type": "array", "items": {"type": "string"}},
            "known_analogues": {"type": "array", "items": {"type": "string"}},
            "llm_origin_assessment": {
                "type": "string",
                "enum": sorted(CLAUDE_ORIGIN_ASSESSMENTS),
            },
            "proof_plan": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
            "representation": {
                "type": "string",
                "enum": sorted(CLAUDE_REPRESENTATIONS),
            },
            "source_idea_domains": {"type": "array", "items": {"type": "string"}},
            "synthesis_note": {"type": "string"},
        },
        "required": [
            "expression",
            "falsifiers",
            "family",
            "hypothesis_id",
            "invariants",
            "known_analogues",
            "llm_origin_assessment",
            "proof_plan",
            "rationale",
            "representation",
            "source_idea_domains",
            "synthesis_note",
        ],
        "additionalProperties": False,
    }
    action = {
        "type": "object",
        "properties": {
            "blocker_kind": {"type": "string"},
            "candidate_id": {"type": "string"},
            "distance_denominator": {"type": "integer"},
            "distance_numerator": {"type": "integer"},
            "repair": {"type": "string"},
            "verdict": {"type": "string", "enum": ["reject", "repair", "retain"]},
        },
        "required": [
            "blocker_kind",
            "candidate_id",
            "distance_denominator",
            "distance_numerator",
            "repair",
            "verdict",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "benchmark_id": {"type": "string", "const": benchmark_id},
            "hypotheses": {
                "type": "array",
                "items": hypothesis,
            },
            "role": {"type": "string", "const": role.value},
            "schema_version": {"type": "string", "const": CLAUDE_OUTPUT_SCHEMA_VERSION},
            "steering_actions": {
                "type": "array",
                "items": action,
            },
        },
        "required": ["benchmark_id", "hypotheses", "role", "schema_version", "steering_actions"],
        "additionalProperties": False,
    }


class ClaudeCreativityClient:
    def __init__(self, config: ClaudeAPIConfig, transport: Transport = urllib_transport) -> None:
        self.config = config
        self.transport = transport
        self.budget = ClaudeBudget()
        self._model_evidence: Mapping[str, Any] | None = None

    def _credential(self) -> str | None:
        value = os.environ.get(self.config.credential_env_var)
        return value.strip() if value and value.strip() else None

    def _headers(self, credential: str) -> dict[str, str]:
        return {
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "x-api-key": credential,
        }

    def _verify_model(self, credential: str) -> Mapping[str, Any]:
        if self._model_evidence is not None:
            return self._model_evidence
        url = f"{MODELS_ENDPOINT}/{urllib.parse.quote(self.config.model, safe='')}"
        status, payload = self.transport(
            "GET", url, self._headers(credential), None, float(self.config.timeout_seconds)
        )
        capabilities = payload.get("capabilities", {})
        structured = capabilities.get("structured_outputs", {}) if isinstance(capabilities, Mapping) else {}
        if (
            status != 200
            or payload.get("id") != self.config.model
            or not isinstance(structured, Mapping)
            or structured.get("supported") is not True
        ):
            raise ClaudeCreativityError("configured Claude model lacks structured-output evidence")
        self._model_evidence = {
            "capabilities_sha256": canonical_sha256(capabilities),
            "model": self.config.model,
            "structured_outputs_supported": True,
        }
        return self._model_evidence

    def _check_budget(self) -> None:
        if self.budget.calls >= self.config.maximum_calls:
            raise ClaudeCreativityError("Claude call budget exhausted")
        if self.budget.total_tokens >= self.config.maximum_total_tokens:
            raise ClaudeCreativityError("Claude token budget exhausted")

    def run(
        self,
        role: ClaudeRole,
        benchmark_id: str,
        public_payload: Mapping[str, Any],
        *,
        candidate_summaries: Sequence[Mapping[str, Any]] = (),
    ) -> ClaudeCallResult:
        benchmark_id = _identifier(benchmark_id, "Claude benchmark_id")
        if _contains_forbidden_key(public_payload):
            raise ClaudeCreativityError("Claude public payload contains sealed target material")
        if role in {ClaudeRole.PROPOSER, ClaudeRole.ANALOGUE_SCOUT} and candidate_summaries:
            raise ClaudeCreativityError("blind Claude creative role cannot see post-generation candidates")
        if role is ClaudeRole.CRITIC and not candidate_summaries:
            raise ClaudeCreativityError("Claude critic requires candidate summaries")
        if not self.config.execution_enabled:
            return ClaudeCallResult(
                ClaudeCallStatus.BLOCKED_DISABLED,
                role,
                benchmark_id,
                None,
                {"credential_persisted": False, "network_calls": 0},
            )
        credential = self._credential()
        if credential is None:
            return ClaudeCallResult(
                ClaudeCallStatus.BLOCKED_MISSING_CREDENTIAL,
                role,
                benchmark_id,
                None,
                {
                    "credential_env_var": self.config.credential_env_var,
                    "credential_persisted": False,
                    "network_calls": 0,
                },
            )
        self._check_budget()
        model_evidence = self._verify_model(credential)
        instructions = {
            ClaudeRole.PROPOSER: (
                "Propose structurally distinct mathematical hypotheses and proof plans. For "
                "each idea, self-assess whether it is a known rewrite, cross-domain synthesis, "
                "proposed new construction, or uncertain; name analogues and source domains. "
                "Uncertainty is welcome and no idea is pruned by this label."
            ),
            ClaudeRole.CRITIC: (
                "Critique candidates, identify typed blockers, and suggest repairs or "
                "recombinations without treating a failed check as grounds to delete an idea."
            ),
            ClaudeRole.ANALOGUE_SCOUT: (
                "Scout distant mathematical analogues, name the source domains, and turn each "
                "analogy into a typed hypothesis. Mark uncertain lineage explicitly."
            ),
            ClaudeRole.DATASET_EXPLAINER: (
                "Propose multiple structural explanations for the public dataset, including "
                "invariants, confounders, shift sensitivity, and falsifying interventions."
            ),
            ClaudeRole.PROOF_STRATEGIST: (
                "Propose proof routes independent of the candidate's declared plan, varying "
                "induction variables, invariants, bijections, descent, transforms, and contradiction."
            ),
            ClaudeRole.RECOMBINER: (
                "Mix retained ideas across source domains and representations. Preserve parent "
                "lineage, and label known rewrites, syntheses, proposed constructions, and uncertainty."
            ),
            ClaudeRole.REPRESENTATION_INVENTOR: (
                "Invent typed representations that make the structure easier to express or test, "
                "including recurrences, generating functions, sums, products, modular, tensor, "
                "transform, and variational forms."
            ),
        }
        prompt_body = {
            "benchmark": public_payload,
            "candidate_summaries": [dict(item) for item in candidate_summaries],
            "instruction": instructions[role],
            "role": role.value,
        }
        prompt = json.dumps(prompt_body, sort_keys=True, separators=(",", ":"))
        schema = _structured_output_schema(role, benchmark_id)
        request_body = {
            "max_tokens": self.config.maximum_output_tokens,
            "messages": [{"content": prompt, "role": "user"}],
            "model": self.config.model,
            "output_config": {
                "effort": self.config.effort,
                "format": {"schema": schema, "type": "json_schema"},
            },
            "system": (
                "You are one bounded component in a blind mathematical discovery experiment. "
                "Return only the required structured object. You may propose or critique, but "
                "you do not verify correctness or claim novelty. Origin labels are your "
                "fallible self-assessment for lineage tracking, not prior-art conclusions."
            ),
        }
        request_bytes = json.dumps(request_body, sort_keys=True, separators=(",", ":")).encode()
        status, response = self.transport(
            "POST",
            MESSAGES_ENDPOINT,
            self._headers(credential),
            request_bytes,
            float(self.config.timeout_seconds),
        )
        if status != 200 or response.get("type") != "message":
            raise ClaudeCreativityError("Claude Messages API returned an invalid envelope")
        if response.get("stop_reason") != "end_turn" or response.get("role") != "assistant":
            raise ClaudeCreativityError("Claude response did not complete normally")
        content = response.get("content")
        if (
            not isinstance(content, list)
            or len(content) != 1
            or not isinstance(content[0], Mapping)
            or content[0].get("type") != "text"
            or not isinstance(content[0].get("text"), str)
        ):
            raise ClaudeCreativityError("Claude structured response content changed")
        try:
            raw_output = json.loads(content[0]["text"])
        except json.JSONDecodeError as error:
            raise ClaudeCreativityError("Claude structured text is not JSON") from error
        output = ClaudeStructuredOutput.from_mapping(raw_output)
        if output.role is not role or output.benchmark_id != benchmark_id:
            raise ClaudeCreativityError("Claude output crossed its requested role or benchmark")
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise ClaudeCreativityError("Claude response omitted usage evidence")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if (
            isinstance(input_tokens, bool)
            or isinstance(output_tokens, bool)
            or not isinstance(input_tokens, int)
            or not isinstance(output_tokens, int)
        ):
            raise ClaudeCreativityError("Claude usage token counts changed")
        next_budget = self.budget.record(input_tokens, output_tokens)
        if next_budget.total_tokens > self.config.maximum_total_tokens:
            raise ClaudeCreativityError("Claude response exceeded the total token budget")
        self.budget = next_budget
        output_dict = output.to_dict()
        evidence = {
            "anthropic_version": ANTHROPIC_VERSION,
            "api_response_id": _identifier(response.get("id"), "Claude response id"),
            "credential_persisted": False,
            "header_names": ["anthropic-version", "content-type", "x-api-key"],
            "model": response.get("model"),
            "model_evidence": dict(model_evidence),
            "network_calls": 2 if self.budget.calls == 1 else 1,
            "output_sha256": canonical_sha256(output_dict),
            "raw_output_sha256": canonical_sha256(raw_output),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "request_schema_sha256": canonical_sha256(schema),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
        return ClaudeCallResult(
            ClaudeCallStatus.COMPLETED,
            role,
            benchmark_id,
            output,
            evidence,
        )
