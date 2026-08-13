"""Provider-neutral, secret-safe LLM proposals for Sigma Core.

The module performs no network access and imports no provider SDK.  A caller injects one callback
which receives a transient request and the *name* of a credential environment variable, never a
credential value.  Raw prompts and response envelopes are not retained in manifests.  Every
successful output remains quarantined and requires the ordinary downstream gates.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sigma_core import (
    ArtifactKind,
    ArtifactRef,
    CandidateArtifact,
    DomainPackRef,
    OutcomeStatus,
    ProvenanceRecord,
    SchemaViolation,
    SourceBinding,
    canonical_json_bytes,
    canonical_sha256,
)

SCHEMA_VERSION = "sigma-llm-candidate-generator-1.0"
RESPONSE_SCHEMA_VERSION = "sigma-provider-neutral-proposals-1.0"
RECEIPT_SCHEMA_VERSION = "sigma-llm-quarantine-receipt-1.0"
LINEAGE_SCHEMA_VERSION = "sigma-llm-proposal-lineage-1.0"

HARD_MAXIMUM_TOTAL_MICRO_USD = 1_000_000_000_000
HARD_MAXIMUM_CALLS = 64
HARD_MAXIMUM_PROMPT_TOKENS = 1_000_000
HARD_MAXIMUM_COMPLETION_TOKENS = 1_000_000
HARD_MAXIMUM_RESPONSE_BYTES = 10_000_000
HARD_MAXIMUM_PROPOSALS = 128
HARD_MAXIMUM_PROMPT_BYTES = 1_000_000

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_ENV_VAR = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_ALLOWED_KINDS = {
    ArtifactKind.ALGORITHM,
    ArtifactKind.CONJECTURE,
    ArtifactKind.CONSTRUCTION,
    ArtifactKind.FORMULA,
    ArtifactKind.PHYSICAL_ACTION,
}
_BOUNDARY_ASSUMPTIONS = (
    "LLM generation is proposal-only and establishes no truth, novelty, or promotion.",
    "Ordinary downstream domain gates remain mandatory.",
)
_FIXED_CLAIMS = ("llm_generated_proposal", "requires_downstream_gates")
_SOURCE_PATHS = {
    "adapter": "src/sigma_theory_compiler/llm_candidate_generator.py",
    "legacy_safety_concepts": "src/sigma_theory_compiler/llm_formula_proposal_adapter.py",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SchemaViolation(f"{label} keys changed")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} is not a canonical identifier")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaViolation(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise SchemaViolation(f"{label} exceeds hard maximum {maximum}")
    return value


def _reasons(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_identifier(item, "reason code") for item in values)
    if not result or result != tuple(sorted(set(result))):
        raise SchemaViolation("reason codes must be nonempty, unique, and sorted")
    return result


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SchemaViolation(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class LLMPolicy:
    provider_id: str
    credential_env_var: str
    maximum_total_micro_usd: int
    maximum_call_micro_usd: int
    maximum_calls: int
    maximum_prompt_tokens: int
    maximum_completion_tokens: int
    maximum_response_bytes: int
    maximum_proposals: int

    def __post_init__(self) -> None:
        _identifier(self.provider_id, "provider_id")
        if (
            not isinstance(self.credential_env_var, str)
            or _ENV_VAR.fullmatch(self.credential_env_var) is None
        ):
            raise SchemaViolation("credential_env_var must be an environment-variable name")
        _integer(
            self.maximum_total_micro_usd,
            "maximum_total_micro_usd",
            minimum=1,
            maximum=HARD_MAXIMUM_TOTAL_MICRO_USD,
        )
        _integer(
            self.maximum_call_micro_usd,
            "maximum_call_micro_usd",
            minimum=1,
            maximum=self.maximum_total_micro_usd,
        )
        _integer(
            self.maximum_calls,
            "maximum_calls",
            minimum=1,
            maximum=HARD_MAXIMUM_CALLS,
        )
        _integer(
            self.maximum_prompt_tokens,
            "maximum_prompt_tokens",
            minimum=1,
            maximum=HARD_MAXIMUM_PROMPT_TOKENS,
        )
        _integer(
            self.maximum_completion_tokens,
            "maximum_completion_tokens",
            minimum=1,
            maximum=HARD_MAXIMUM_COMPLETION_TOKENS,
        )
        _integer(
            self.maximum_response_bytes,
            "maximum_response_bytes",
            minimum=1,
            maximum=HARD_MAXIMUM_RESPONSE_BYTES,
        )
        _integer(
            self.maximum_proposals,
            "maximum_proposals",
            minimum=1,
            maximum=HARD_MAXIMUM_PROPOSALS,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "credential_env_var": self.credential_env_var,
            "maximum_total_micro_usd": self.maximum_total_micro_usd,
            "maximum_call_micro_usd": self.maximum_call_micro_usd,
            "maximum_calls": self.maximum_calls,
            "maximum_prompt_tokens": self.maximum_prompt_tokens,
            "maximum_completion_tokens": self.maximum_completion_tokens,
            "maximum_response_bytes": self.maximum_response_bytes,
            "maximum_proposals": self.maximum_proposals,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LLMPolicy:
        expected = (
            "provider_id",
            "credential_env_var",
            "maximum_total_micro_usd",
            "maximum_call_micro_usd",
            "maximum_calls",
            "maximum_prompt_tokens",
            "maximum_completion_tokens",
            "maximum_response_bytes",
            "maximum_proposals",
        )
        _exact_keys(value, set(expected), "LLM policy")
        return cls(*(value[key] for key in expected))


@dataclass(frozen=True, slots=True)
class LLMBudgetState:
    calls: int = 0
    spent_micro_usd: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("calls", "spent_micro_usd", "prompt_tokens", "completion_tokens"):
            _integer(getattr(self, name), name)

    def record_call(self, usage: LLMUsage | None = None) -> LLMBudgetState:
        return LLMBudgetState(
            self.calls + 1,
            self.spent_micro_usd + (0 if usage is None else usage.billed_micro_usd),
            self.prompt_tokens + (0 if usage is None else usage.prompt_tokens),
            self.completion_tokens + (0 if usage is None else usage.completion_tokens),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "spent_micro_usd": self.spent_micro_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LLMBudgetState:
        expected = ("calls", "spent_micro_usd", "prompt_tokens", "completion_tokens")
        _exact_keys(value, set(expected), "LLM budget state")
        return cls(*(value[key] for key in expected))


@dataclass(frozen=True, slots=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    billed_micro_usd: int

    def __post_init__(self) -> None:
        _integer(self.prompt_tokens, "prompt_tokens")
        _integer(self.completion_tokens, "completion_tokens")
        _integer(self.billed_micro_usd, "billed_micro_usd")

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "billed_micro_usd": self.billed_micro_usd,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LLMUsage:
        expected = ("prompt_tokens", "completion_tokens", "billed_micro_usd")
        _exact_keys(value, set(expected), "LLM usage")
        return cls(*(value[key] for key in expected))


@dataclass(frozen=True, slots=True)
class LLMProposalRequest:
    request_id: str
    prompt: str
    prompt_token_count: int
    completion_token_limit: int
    deterministic_seed: int
    context: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        if (
            not isinstance(self.prompt, str)
            or not self.prompt.strip()
            or self.prompt != self.prompt.strip()
            or len(self.prompt.encode("utf-8")) > HARD_MAXIMUM_PROMPT_BYTES
        ):
            raise SchemaViolation("prompt must be nonempty, stripped, and within its hard byte cap")
        _integer(
            self.prompt_token_count,
            "prompt_token_count",
            minimum=1,
            maximum=HARD_MAXIMUM_PROMPT_TOKENS,
        )
        _integer(
            self.completion_token_limit,
            "completion_token_limit",
            minimum=1,
            maximum=HARD_MAXIMUM_COMPLETION_TOKENS,
        )
        _integer(self.deterministic_seed, "deterministic_seed", maximum=2**63 - 1)
        if self.context != tuple(sorted(self.context, key=lambda item: item.artifact_id)):
            raise SchemaViolation("request context must be sorted by artifact_id")
        if len({item.artifact_id for item in self.context}) != len(self.context):
            raise SchemaViolation("request context contains duplicate artifact IDs")

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def contract_dict(self) -> dict[str, Any]:
        """Return the persistable request contract, intentionally excluding the prompt body."""

        return {
            "request_id": self.request_id,
            "prompt_sha256": self.prompt_sha256,
            "prompt_bytes": len(self.prompt.encode("utf-8")),
            "prompt_token_count": self.prompt_token_count,
            "completion_token_limit": self.completion_token_limit,
            "deterministic_seed": self.deterministic_seed,
            "context": [item.to_dict() for item in self.context],
        }

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.contract_dict())


@dataclass(frozen=True, slots=True)
class ProposalLineage:
    ordinal: int
    provider_proposal_id: str
    proposal_sha256: str
    candidate: ArtifactRef
    first_ordinal: int
    duplicate: bool
    lineage_sha256: str
    schema_version: str = LINEAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LINEAGE_SCHEMA_VERSION:
            raise SchemaViolation("proposal lineage schema_version changed")
        _integer(self.ordinal, "lineage ordinal")
        _identifier(self.provider_proposal_id, "provider proposal ID")
        _sha(self.proposal_sha256, "proposal_sha256")
        _integer(self.first_ordinal, "lineage first_ordinal")
        if not isinstance(self.duplicate, bool):
            raise SchemaViolation("lineage duplicate must be boolean")
        if (
            self.duplicate != (self.first_ordinal != self.ordinal)
            or self.first_ordinal > self.ordinal
        ):
            raise SchemaViolation("proposal duplicate lineage changed")
        if self.lineage_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("proposal lineage hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "provider_proposal_id": self.provider_proposal_id,
            "proposal_sha256": self.proposal_sha256,
            "candidate": self.candidate.to_dict(),
            "first_ordinal": self.first_ordinal,
            "duplicate": self.duplicate,
        }

    @classmethod
    def create(
        cls,
        ordinal: int,
        provider_proposal_id: str,
        proposal_sha256: str,
        candidate: ArtifactRef,
        first_ordinal: int,
    ) -> ProposalLineage:
        body = {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "ordinal": ordinal,
            "provider_proposal_id": provider_proposal_id,
            "proposal_sha256": proposal_sha256,
            "candidate": candidate.to_dict(),
            "first_ordinal": first_ordinal,
            "duplicate": first_ordinal != ordinal,
        }
        return cls(
            ordinal,
            provider_proposal_id,
            proposal_sha256,
            candidate,
            first_ordinal,
            first_ordinal != ordinal,
            canonical_sha256(body),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "lineage_sha256": self.lineage_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProposalLineage:
        expected = (
            "schema_version",
            "ordinal",
            "provider_proposal_id",
            "proposal_sha256",
            "candidate",
            "first_ordinal",
            "duplicate",
            "lineage_sha256",
        )
        _exact_keys(value, set(expected), "proposal lineage")
        return cls(
            value["ordinal"],
            str(value["provider_proposal_id"]),
            str(value["proposal_sha256"]),
            ArtifactRef.from_dict(value["candidate"]),
            value["first_ordinal"],
            value["duplicate"],
            str(value["lineage_sha256"]),
            str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class QuarantineReceipt:
    status: OutcomeStatus
    reason_codes: tuple[str, ...]
    provider_id: str
    credential_env_var: str
    policy_sha256: str
    request_sha256: str
    response_sha256: str | None
    response_bytes: int
    usage: LLMUsage | None
    budget_before: LLMBudgetState
    budget_after: LLMBudgetState
    call_recorded: bool
    charge_applied: bool
    proposal_count: int
    unique_count: int
    duplicate_count: int
    receipt_sha256: str
    schema_version: str = RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_VERSION:
            raise SchemaViolation("quarantine receipt schema_version changed")
        _identifier(self.provider_id, "receipt provider_id")
        if _ENV_VAR.fullmatch(self.credential_env_var) is None:
            raise SchemaViolation("receipt credential reference changed")
        _sha(self.policy_sha256, "receipt policy_sha256")
        _sha(self.request_sha256, "receipt request_sha256")
        if self.response_sha256 is not None:
            _sha(self.response_sha256, "receipt response_sha256")
        _integer(self.response_bytes, "receipt response_bytes")
        for name in ("proposal_count", "unique_count", "duplicate_count"):
            _integer(getattr(self, name), name)
        if self.proposal_count != self.unique_count + self.duplicate_count:
            raise SchemaViolation("receipt proposal accounting changed")
        if not isinstance(self.call_recorded, bool) or not isinstance(self.charge_applied, bool):
            raise SchemaViolation("receipt call/charge accounting flags must be boolean")
        if self.charge_applied and (not self.call_recorded or self.usage is None):
            raise SchemaViolation("receipt charge requires a recorded call and exact usage")
        expected_after = (
            self.budget_before.record_call(self.usage if self.charge_applied else None)
            if self.call_recorded
            else self.budget_before
        )
        if self.budget_after != expected_after:
            raise SchemaViolation("receipt call/charge accounting changed")
        if self.status is OutcomeStatus.PASS:
            if (
                self.reason_codes
                or self.usage is None
                or not self.call_recorded
                or not self.charge_applied
            ):
                raise SchemaViolation("pass receipt boundary changed")
            if self.response_sha256 is None or self.proposal_count < 1:
                raise SchemaViolation("pass receipt lacks response evidence")
        else:
            _reasons(self.reason_codes)
            if self.proposal_count or self.unique_count or self.duplicate_count:
                raise SchemaViolation("non-pass receipt cannot admit proposals")
        if self.receipt_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("quarantine receipt hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "provider_id": self.provider_id,
            "credential_env_var": self.credential_env_var,
            "policy_sha256": self.policy_sha256,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "response_bytes": self.response_bytes,
            "usage": None if self.usage is None else self.usage.to_dict(),
            "budget_before": self.budget_before.to_dict(),
            "budget_after": self.budget_after.to_dict(),
            "call_recorded": self.call_recorded,
            "charge_applied": self.charge_applied,
            "proposal_count": self.proposal_count,
            "unique_count": self.unique_count,
            "duplicate_count": self.duplicate_count,
            "quarantined": True,
            "request_body_persisted": False,
            "response_body_persisted": False,
            "credential_value_accessed": False,
            "truth_established": False,
            "novelty_established": False,
            "promotion_allowed": False,
            "downstream_gates_required": True,
        }

    @classmethod
    def create(
        cls,
        *,
        status: OutcomeStatus,
        reason_codes: Sequence[str],
        policy: LLMPolicy,
        request: LLMProposalRequest,
        response_sha256: str | None,
        response_bytes: int,
        usage: LLMUsage | None,
        budget_before: LLMBudgetState,
        budget_after: LLMBudgetState,
        call_recorded: bool,
        charge_applied: bool,
        proposal_count: int = 0,
        unique_count: int = 0,
        duplicate_count: int = 0,
    ) -> QuarantineReceipt:
        reasons = tuple(sorted(reason_codes))
        fields = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": status.value,
            "reason_codes": list(reasons),
            "provider_id": policy.provider_id,
            "credential_env_var": policy.credential_env_var,
            "policy_sha256": canonical_sha256(policy.to_dict()),
            "request_sha256": request.request_sha256,
            "response_sha256": response_sha256,
            "response_bytes": response_bytes,
            "usage": None if usage is None else usage.to_dict(),
            "budget_before": budget_before.to_dict(),
            "budget_after": budget_after.to_dict(),
            "call_recorded": call_recorded,
            "charge_applied": charge_applied,
            "proposal_count": proposal_count,
            "unique_count": unique_count,
            "duplicate_count": duplicate_count,
            "quarantined": True,
            "request_body_persisted": False,
            "response_body_persisted": False,
            "credential_value_accessed": False,
            "truth_established": False,
            "novelty_established": False,
            "promotion_allowed": False,
            "downstream_gates_required": True,
        }
        return cls(
            status,
            reasons,
            policy.provider_id,
            policy.credential_env_var,
            fields["policy_sha256"],
            request.request_sha256,
            response_sha256,
            response_bytes,
            usage,
            budget_before,
            budget_after,
            call_recorded,
            charge_applied,
            proposal_count,
            unique_count,
            duplicate_count,
            canonical_sha256(fields),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QuarantineReceipt:
        expected = {
            "schema_version",
            "status",
            "reason_codes",
            "provider_id",
            "credential_env_var",
            "policy_sha256",
            "request_sha256",
            "response_sha256",
            "response_bytes",
            "usage",
            "budget_before",
            "budget_after",
            "call_recorded",
            "charge_applied",
            "proposal_count",
            "unique_count",
            "duplicate_count",
            "quarantined",
            "request_body_persisted",
            "response_body_persisted",
            "credential_value_accessed",
            "truth_established",
            "novelty_established",
            "promotion_allowed",
            "downstream_gates_required",
            "receipt_sha256",
        }
        _exact_keys(value, expected, "quarantine receipt")
        if (
            value["quarantined"] is not True
            or value["request_body_persisted"] is not False
            or value["response_body_persisted"] is not False
            or value["credential_value_accessed"] is not False
            or value["truth_established"] is not False
            or value["novelty_established"] is not False
            or value["promotion_allowed"] is not False
            or value["downstream_gates_required"] is not True
            or not isinstance(value["reason_codes"], list)
        ):
            raise SchemaViolation("quarantine scientific or persistence boundary changed")
        try:
            status = OutcomeStatus(value["status"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("quarantine status is not registered") from error
        usage = None if value["usage"] is None else LLMUsage.from_dict(value["usage"])
        return cls(
            status,
            tuple(value["reason_codes"]),
            str(value["provider_id"]),
            str(value["credential_env_var"]),
            str(value["policy_sha256"]),
            str(value["request_sha256"]),
            None if value["response_sha256"] is None else str(value["response_sha256"]),
            value["response_bytes"],
            usage,
            LLMBudgetState.from_dict(value["budget_before"]),
            LLMBudgetState.from_dict(value["budget_after"]),
            value["call_recorded"],
            value["charge_applied"],
            value["proposal_count"],
            value["unique_count"],
            value["duplicate_count"],
            str(value["receipt_sha256"]),
            str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class LLMProposalManifest:
    policy: LLMPolicy
    domain_pack: DomainPackRef
    request_contract: Mapping[str, Any]
    sources: tuple[SourceBinding, ...]
    receipt: QuarantineReceipt
    candidates: tuple[CandidateArtifact, ...]
    lineage: tuple[ProposalLineage, ...]
    manifest_sha256: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaViolation("LLM manifest schema_version changed")
        expected_request_keys = {
            "request_id",
            "prompt_sha256",
            "prompt_bytes",
            "prompt_token_count",
            "completion_token_limit",
            "deterministic_seed",
            "context",
        }
        _exact_keys(self.request_contract, expected_request_keys, "LLM request contract")
        object.__setattr__(
            self,
            "request_contract",
            json.loads(canonical_json_bytes(self.request_contract).decode("utf-8")),
        )
        if "prompt" in self.request_contract:
            raise SchemaViolation("prompt body entered persisted request contract")
        if self.sources != tuple(sorted(self.sources, key=lambda item: item.role)):
            raise SchemaViolation("LLM manifest sources must be sorted")
        if {item.role for item in self.sources} != set(_SOURCE_PATHS):
            raise SchemaViolation("LLM manifest source roles changed")
        if self.receipt.policy_sha256 != canonical_sha256(self.policy.to_dict()):
            raise SchemaViolation("LLM receipt policy binding changed")
        if self.receipt.request_sha256 != canonical_sha256(self.request_contract):
            raise SchemaViolation("LLM receipt request binding changed")
        if self.receipt.status is OutcomeStatus.PASS:
            if not self.candidates or not self.lineage:
                raise SchemaViolation("pass LLM manifest has no quarantined candidates")
            if len(self.candidates) != self.receipt.unique_count:
                raise SchemaViolation("LLM unique candidate count changed")
            if len(self.lineage) != self.receipt.proposal_count:
                raise SchemaViolation("LLM proposal lineage count changed")
        elif self.candidates or self.lineage:
            raise SchemaViolation("non-pass LLM manifest emitted partial candidates")
        if tuple(item.ordinal for item in self.lineage) != tuple(range(len(self.lineage))):
            raise SchemaViolation("LLM lineage ordinals are not contiguous")
        refs = {item.artifact_id: item.ref for item in self.candidates}
        if len(refs) != len(self.candidates):
            raise SchemaViolation("LLM candidates contain duplicate artifact IDs")
        first_seen: dict[str, ProposalLineage] = {}
        for record in self.lineage:
            if (
                record.candidate.artifact_id not in refs
                or refs[record.candidate.artifact_id] != record.candidate
            ):
                raise SchemaViolation("LLM lineage candidate is outside the manifest")
            prior = first_seen.setdefault(record.proposal_sha256, record)
            if record.first_ordinal != prior.ordinal or record.candidate != prior.candidate:
                raise SchemaViolation("LLM canonical dedup lineage changed")
        expected_candidate_order = tuple(
            record.candidate.artifact_id for record in self.lineage if not record.duplicate
        )
        if tuple(item.artifact_id for item in self.candidates) != expected_candidate_order:
            raise SchemaViolation("LLM candidates are not in canonical first-seen order")
        for candidate in self.candidates:
            candidate.validate()
            if candidate.provenance.domain_pack != self.domain_pack:
                raise SchemaViolation("LLM candidate domain-pack binding changed")
            if candidate.claims != _FIXED_CLAIMS:
                raise SchemaViolation("LLM candidate claims escaped quarantine")
        if self.manifest_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("LLM manifest hash changed")

    @property
    def status(self) -> OutcomeStatus:
        return self.receipt.status

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.to_dict(),
            "domain_pack": self.domain_pack.to_dict(),
            "request_contract": dict(self.request_contract),
            "sources": [item.to_dict() for item in self.sources],
            "receipt": self.receipt.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "lineage": [item.to_dict() for item in self.lineage],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "manifest_sha256": self.manifest_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LLMProposalManifest:
        expected = {
            "schema_version",
            "policy",
            "domain_pack",
            "request_contract",
            "sources",
            "receipt",
            "candidates",
            "lineage",
            "manifest_sha256",
        }
        _exact_keys(value, expected, "LLM proposal manifest")
        for key in ("sources", "candidates", "lineage"):
            if not isinstance(value[key], list):
                raise SchemaViolation(f"LLM manifest {key} must be an array")
        return cls(
            LLMPolicy.from_dict(value["policy"]),
            DomainPackRef.from_dict(value["domain_pack"]),
            value["request_contract"],
            tuple(SourceBinding.from_dict(item) for item in value["sources"]),
            QuarantineReceipt.from_dict(value["receipt"]),
            tuple(CandidateArtifact.from_dict(item) for item in value["candidates"]),
            tuple(ProposalLineage.from_dict(item) for item in value["lineage"]),
            str(value["manifest_sha256"]),
            str(value["schema_version"]),
        )


ProviderCallback = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _checked_sources(sources: Sequence[SourceBinding]) -> tuple[SourceBinding, ...]:
    ordered = tuple(sorted(sources, key=lambda item: item.role))
    if len(ordered) != len(_SOURCE_PATHS) or {item.role for item in ordered} != set(_SOURCE_PATHS):
        raise SchemaViolation("source bindings must contain adapter and legacy_safety_concepts")
    if any(item.path != _SOURCE_PATHS[item.role] for item in ordered):
        raise SchemaViolation("LLM source binding path changed")
    return ordered


def llm_source_bindings(project_root: str | Path) -> tuple[SourceBinding, ...]:
    root = Path(project_root).resolve()
    result = []
    for role, relative in sorted(_SOURCE_PATHS.items()):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SchemaViolation("LLM source escaped project root") from error
        if not path.is_file():
            raise SchemaViolation(f"LLM source is not a file: {relative}")
        result.append(SourceBinding(role, relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(result)


def _receipt_manifest(
    policy: LLMPolicy,
    domain_pack: DomainPackRef,
    request: LLMProposalRequest,
    sources: tuple[SourceBinding, ...],
    budget: LLMBudgetState,
    status: OutcomeStatus,
    reasons: Sequence[str],
    *,
    response_sha256: str | None = None,
    response_bytes: int = 0,
    usage: LLMUsage | None = None,
    call_recorded: bool = False,
    charge_applied: bool = False,
    candidates: Sequence[CandidateArtifact] = (),
    lineage: Sequence[ProposalLineage] = (),
) -> LLMProposalManifest:
    if status is OutcomeStatus.PASS:
        call_recorded = True
        charge_applied = True
    budget_after = (
        budget.record_call(usage if charge_applied else None) if call_recorded else budget
    )
    receipt = QuarantineReceipt.create(
        status=status,
        reason_codes=reasons,
        policy=policy,
        request=request,
        response_sha256=response_sha256,
        response_bytes=response_bytes,
        usage=usage,
        budget_before=budget,
        budget_after=budget_after,
        call_recorded=call_recorded,
        charge_applied=charge_applied,
        proposal_count=len(lineage) if status is OutcomeStatus.PASS else 0,
        unique_count=len(candidates) if status is OutcomeStatus.PASS else 0,
        duplicate_count=(len(lineage) - len(candidates) if status is OutcomeStatus.PASS else 0),
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy.to_dict(),
        "domain_pack": domain_pack.to_dict(),
        "request_contract": request.contract_dict(),
        "sources": [item.to_dict() for item in sources],
        "receipt": receipt.to_dict(),
        "candidates": [item.to_dict() for item in candidates],
        "lineage": [item.to_dict() for item in lineage],
    }
    return LLMProposalManifest(
        policy,
        domain_pack,
        request.contract_dict(),
        sources,
        receipt,
        tuple(candidates),
        tuple(lineage),
        canonical_sha256(body),
    )


def _normalized_proposals(
    response: Mapping[str, Any], policy: LLMPolicy, request: LLMProposalRequest
) -> tuple[LLMUsage, list[dict[str, Any]]]:
    _exact_keys(
        response, {"schema_version", "request_id", "usage", "proposals"}, "provider response"
    )
    if (
        response["schema_version"] != RESPONSE_SCHEMA_VERSION
        or response["request_id"] != request.request_id
    ):
        raise SchemaViolation("provider response schema or request binding changed")
    if (
        not isinstance(response["proposals"], list)
        or not 1 <= len(response["proposals"]) <= policy.maximum_proposals
    ):
        raise SchemaViolation("provider proposal count is outside the configured cap")
    usage = LLMUsage.from_dict(response["usage"])
    normalized = []
    seen_ids = set()
    for raw in response["proposals"]:
        _exact_keys(
            raw,
            {"proposal_id", "kind", "statement", "representation", "assumptions"},
            "provider proposal",
        )
        proposal_id = _identifier(raw["proposal_id"], "provider proposal ID")
        if proposal_id in seen_ids:
            raise SchemaViolation("provider proposal IDs contain duplicates")
        seen_ids.add(proposal_id)
        try:
            kind = ArtifactKind(raw["kind"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("provider proposal kind is not registered") from error
        if kind not in _ALLOWED_KINDS:
            raise SchemaViolation("provider proposal kind requires proof rather than quarantine")
        statement = raw["statement"]
        if (
            not isinstance(statement, str)
            or not statement.strip()
            or statement != statement.strip()
            or len(statement.encode("utf-8")) > 16_384
        ):
            raise SchemaViolation("provider proposal statement is malformed or too large")
        assumptions = raw["assumptions"]
        if not isinstance(assumptions, list) or len(assumptions) > 64:
            raise SchemaViolation("provider proposal assumptions are malformed or too numerous")
        assumptions = tuple(assumptions)
        if any(
            not isinstance(item, str) or not item.strip() or item != item.strip()
            for item in assumptions
        ):
            raise SchemaViolation("provider proposal assumption is malformed")
        if assumptions != tuple(sorted(set(assumptions))):
            raise SchemaViolation("provider proposal assumptions must be unique and sorted")
        representation = raw["representation"]
        if not isinstance(representation, Mapping):
            raise SchemaViolation("provider proposal representation must be an object")
        content = {
            "kind": kind.value,
            "statement": statement,
            "representation": representation,
            "assumptions": list(assumptions),
        }
        proposal_sha256 = canonical_sha256(content)
        normalized.append(
            {
                "proposal_id": proposal_id,
                "kind": kind,
                "statement": statement,
                "representation": representation,
                "assumptions": assumptions,
                "proposal_sha256": proposal_sha256,
            }
        )
    return usage, normalized


def generate_llm_candidates(
    policy: LLMPolicy,
    request: LLMProposalRequest,
    budget: LLMBudgetState,
    domain_pack: DomainPackRef,
    sources: Sequence[SourceBinding],
    provider: ProviderCallback,
) -> LLMProposalManifest:
    """Call an injected provider once and return a quarantined, all-or-nothing manifest."""

    if not all(
        (
            isinstance(policy, LLMPolicy),
            isinstance(request, LLMProposalRequest),
            isinstance(budget, LLMBudgetState),
            isinstance(domain_pack, DomainPackRef),
            callable(provider),
        )
    ):
        raise SchemaViolation("LLM generation requires typed inputs and an injected callback")
    checked_sources = _checked_sources(sources)
    preflight_reasons = []
    if budget.calls >= policy.maximum_calls:
        preflight_reasons.append("call_cap_reached")
    if budget.spent_micro_usd >= policy.maximum_total_micro_usd:
        preflight_reasons.append("total_budget_exhausted")
    elif budget.spent_micro_usd + policy.maximum_call_micro_usd > policy.maximum_total_micro_usd:
        preflight_reasons.append("total_budget_reservation_unavailable")
    if request.prompt_token_count > policy.maximum_prompt_tokens:
        preflight_reasons.append("prompt_token_cap_exceeded")
    if request.completion_token_limit > policy.maximum_completion_tokens:
        preflight_reasons.append("completion_token_request_cap_exceeded")
    if preflight_reasons:
        return _receipt_manifest(
            policy,
            domain_pack,
            request,
            checked_sources,
            budget,
            OutcomeStatus.BLOCK,
            preflight_reasons,
        )

    provider_request = {
        "schema_version": "sigma-provider-neutral-request-1.0",
        "request_id": request.request_id,
        "prompt": request.prompt,
        "prompt_sha256": request.prompt_sha256,
        "prompt_token_count": request.prompt_token_count,
        "completion_token_limit": request.completion_token_limit,
        "deterministic_seed": request.deterministic_seed,
        "context": [item.to_dict() for item in request.context],
        "maximum_call_micro_usd": policy.maximum_call_micro_usd,
        "credential_env_var": policy.credential_env_var,
    }
    try:
        response = provider(provider_request)
    except Exception:  # noqa: BLE001 - arbitrary provider boundary is classified, never persisted
        return _receipt_manifest(
            policy,
            domain_pack,
            request,
            checked_sources,
            budget,
            OutcomeStatus.ERROR,
            ("provider_exception",),
            call_recorded=True,
        )
    if not isinstance(response, Mapping):
        return _receipt_manifest(
            policy,
            domain_pack,
            request,
            checked_sources,
            budget,
            OutcomeStatus.REJECT,
            ("malformed_provider_response",),
            call_recorded=True,
        )
    try:
        response_body = canonical_json_bytes(response)
    except SchemaViolation:
        return _receipt_manifest(
            policy,
            domain_pack,
            request,
            checked_sources,
            budget,
            OutcomeStatus.REJECT,
            ("malformed_provider_response",),
            call_recorded=True,
        )
    response_sha256 = hashlib.sha256(response_body).hexdigest()
    response_bytes = len(response_body)
    if response_bytes > policy.maximum_response_bytes:
        return _receipt_manifest(
            policy,
            domain_pack,
            request,
            checked_sources,
            budget,
            OutcomeStatus.REJECT,
            ("response_byte_cap_exceeded",),
            response_sha256=response_sha256,
            response_bytes=response_bytes,
            call_recorded=True,
        )
    reported_usage = None
    try:
        if isinstance(response.get("usage"), Mapping):
            reported_usage = LLMUsage.from_dict(response["usage"])
    except (SchemaViolation, TypeError):
        reported_usage = None
    try:
        usage, proposals = _normalized_proposals(response, policy, request)
    except (SchemaViolation, TypeError):
        return _receipt_manifest(
            policy,
            domain_pack,
            request,
            checked_sources,
            budget,
            OutcomeStatus.REJECT,
            ("malformed_provider_response",),
            response_sha256=response_sha256,
            response_bytes=response_bytes,
            usage=reported_usage,
            call_recorded=True,
            charge_applied=reported_usage is not None,
        )
    usage_reasons = []
    if usage.prompt_tokens != request.prompt_token_count:
        usage_reasons.append("prompt_token_accounting_mismatch")
    if usage.completion_tokens > request.completion_token_limit:
        usage_reasons.append("completion_token_cap_exceeded")
    if usage.billed_micro_usd > policy.maximum_call_micro_usd:
        usage_reasons.append("call_budget_exceeded")
    if budget.spent_micro_usd + usage.billed_micro_usd > policy.maximum_total_micro_usd:
        usage_reasons.append("total_budget_exceeded")
    if usage_reasons:
        return _receipt_manifest(
            policy,
            domain_pack,
            request,
            checked_sources,
            budget,
            OutcomeStatus.REJECT,
            usage_reasons,
            response_sha256=response_sha256,
            response_bytes=response_bytes,
            usage=usage,
            call_recorded=True,
            charge_applied=True,
        )

    policy_sha256 = canonical_sha256(policy.to_dict())
    by_proposal_sha: dict[str, tuple[int, CandidateArtifact]] = {}
    candidates = []
    lineage_inputs: list[tuple[int, str, str, CandidateArtifact, int]] = []
    for ordinal, proposal in enumerate(proposals):
        proposal_sha256 = proposal["proposal_sha256"]
        existing = by_proposal_sha.get(proposal_sha256)
        if existing is None:
            provenance_parameters = {
                "schema_version": SCHEMA_VERSION,
                "policy_sha256": policy_sha256,
                "request_sha256": request.request_sha256,
                "response_sha256": response_sha256,
                "usage": usage.to_dict(),
                "first_ordinal": ordinal,
                "provider_proposal_id": proposal["proposal_id"],
                "proposal_sha256": proposal_sha256,
                "quarantined": True,
                "truth_established": False,
                "novelty_established": False,
                "promotion_allowed": False,
                "downstream_gates_required": True,
            }
            candidate = CandidateArtifact.create(
                proposal["kind"],
                proposal["statement"],
                {
                    "provider_proposal_id": proposal["proposal_id"],
                    "proposal_sha256": proposal_sha256,
                    "provider_assumptions": list(proposal["assumptions"]),
                    "proposal": proposal["representation"],
                    "quarantined": True,
                },
                ProvenanceRecord.create(
                    domain_pack,
                    provenance_parameters,
                    inputs=request.context,
                    sources=checked_sources,
                ),
                assumptions=tuple(sorted({*proposal["assumptions"], *_BOUNDARY_ASSUMPTIONS})),
                claims=_FIXED_CLAIMS,
            )
            by_proposal_sha[proposal_sha256] = (ordinal, candidate)
            candidates.append(candidate)
        else:
            ordinal_first, candidate = existing
            lineage_inputs.append(
                (ordinal, proposal["proposal_id"], proposal_sha256, candidate, ordinal_first)
            )
            continue
        lineage_inputs.append(
            (ordinal, proposal["proposal_id"], proposal_sha256, candidate, ordinal)
        )
    lineage = tuple(
        ProposalLineage.create(ordinal, proposal_id, proposal_sha256, candidate.ref, first_ordinal)
        for ordinal, proposal_id, proposal_sha256, candidate, first_ordinal in lineage_inputs
    )
    return _receipt_manifest(
        policy,
        domain_pack,
        request,
        checked_sources,
        budget,
        OutcomeStatus.PASS,
        (),
        response_sha256=response_sha256,
        response_bytes=response_bytes,
        usage=usage,
        candidates=candidates,
        lineage=lineage,
    )


def validate_llm_manifest(
    value: LLMProposalManifest | Mapping[str, Any],
    *,
    request: LLMProposalRequest | None = None,
    project_root: str | Path | None = None,
) -> LLMProposalManifest:
    """Validate canonical closure and optionally bind live request and source bytes."""

    parsed = (
        value if isinstance(value, LLMProposalManifest) else LLMProposalManifest.from_dict(value)
    )
    parsed = LLMProposalManifest.from_dict(parsed.to_dict())
    if request is not None and parsed.request_contract != request.contract_dict():
        raise SchemaViolation("LLM manifest request replay changed")
    if project_root is not None and parsed.sources != llm_source_bindings(project_root):
        raise SchemaViolation("LLM manifest source bytes changed")
    if parsed.status is not OutcomeStatus.PASS:
        return parsed

    lineage_by_candidate = {
        item.candidate.artifact_id: item for item in parsed.lineage if not item.duplicate
    }
    context = tuple(ArtifactRef.from_dict(item) for item in parsed.request_contract["context"])
    for candidate in parsed.candidates:
        record = lineage_by_candidate[candidate.artifact_id]
        representation = candidate.representation
        expected_keys = {
            "provider_proposal_id",
            "proposal_sha256",
            "provider_assumptions",
            "proposal",
            "quarantined",
        }
        _exact_keys(representation, expected_keys, "LLM candidate representation")
        provider_assumptions = representation["provider_assumptions"]
        if not isinstance(provider_assumptions, list) or representation["quarantined"] is not True:
            raise SchemaViolation("LLM candidate quarantine representation changed")
        proposal_body = {
            "kind": candidate.kind.value,
            "statement": candidate.statement,
            "representation": representation["proposal"],
            "assumptions": provider_assumptions,
        }
        proposal_sha256 = canonical_sha256(proposal_body)
        if (
            proposal_sha256 != representation["proposal_sha256"]
            or proposal_sha256 != record.proposal_sha256
            or representation["provider_proposal_id"] != record.provider_proposal_id
        ):
            raise SchemaViolation("LLM proposal content binding changed")
        parameters = {
            "schema_version": SCHEMA_VERSION,
            "policy_sha256": parsed.receipt.policy_sha256,
            "request_sha256": parsed.receipt.request_sha256,
            "response_sha256": parsed.receipt.response_sha256,
            "usage": parsed.receipt.usage.to_dict(),
            "first_ordinal": record.first_ordinal,
            "provider_proposal_id": record.provider_proposal_id,
            "proposal_sha256": proposal_sha256,
            "quarantined": True,
            "truth_established": False,
            "novelty_established": False,
            "promotion_allowed": False,
            "downstream_gates_required": True,
        }
        expected = CandidateArtifact.create(
            candidate.kind,
            candidate.statement,
            representation,
            ProvenanceRecord.create(
                parsed.domain_pack,
                parameters,
                inputs=context,
                sources=parsed.sources,
            ),
            assumptions=tuple(sorted({*provider_assumptions, *_BOUNDARY_ASSUMPTIONS})),
            claims=_FIXED_CLAIMS,
        )
        if expected.to_dict() != candidate.to_dict():
            raise SchemaViolation("LLM candidate does not match deterministic replay")
    return parsed


__all__ = [
    "HARD_MAXIMUM_CALLS",
    "HARD_MAXIMUM_COMPLETION_TOKENS",
    "HARD_MAXIMUM_PROMPT_BYTES",
    "HARD_MAXIMUM_PROMPT_TOKENS",
    "HARD_MAXIMUM_PROPOSALS",
    "HARD_MAXIMUM_RESPONSE_BYTES",
    "HARD_MAXIMUM_TOTAL_MICRO_USD",
    "LINEAGE_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "LLMBudgetState",
    "LLMPolicy",
    "LLMProposalManifest",
    "LLMProposalRequest",
    "LLMUsage",
    "ProposalLineage",
    "QuarantineReceipt",
    "generate_llm_candidates",
    "llm_source_bindings",
    "validate_llm_manifest",
]
