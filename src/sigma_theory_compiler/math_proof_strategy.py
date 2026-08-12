"""Typed proof-strategy proposals with deterministic, fail-closed execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .math_expression_ir import Equation, Recurrence
from .math_proof import ProofFailure, UnsupportedProof, prove_induction, prove_rational_identity
from .sigma_core import ArtifactRef, SourceBinding, canonical_sha256

_ID = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class ProofStrategyBoundaryError(ValueError):
    """Raised when a proof-strategy record crosses its closed boundary."""


class ProofStrategy(str, Enum):
    EXACT_ALGEBRA = "exact_algebra"
    INDUCTION = "induction"
    CONTRADICTION = "contradiction"
    SUBSTITUTION = "substitution"
    FACTORIZATION = "factorization"
    GENERATING_FUNCTION = "generating_function"
    BIJECTION = "bijection"
    INVARIANT = "invariant"
    SYMMETRY = "symmetry"
    EXTREMAL = "extremal"
    DESCENT = "descent"
    CHANGE_OF_VARIABLES = "change_of_variables"


class ProposalOrigin(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    HUMAN = "human"


class ProofAttemptStatus(str, Enum):
    PROVED = "proved"
    REFUTED = "refuted"
    BLOCKED = "blocked"
    ERROR = "error"


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ProofStrategyBoundaryError(f"{label} is not a canonical identifier")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ProofStrategyBoundaryError(f"{label} is not a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class ProofStrategyProposal:
    proposal_id: str
    artifact: ArtifactRef
    strategy: ProofStrategy
    origin: ProposalOrigin
    parameters_sha256: str
    source: SourceBinding | None
    proposal_sha256: str
    schema_version: str = "sigma-math-proof-strategy-proposal-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-math-proof-strategy-proposal-1.0":
            raise ProofStrategyBoundaryError("proposal schema changed")
        _identifier(self.proposal_id, "proposal ID")
        if not isinstance(self.strategy, ProofStrategy) or not isinstance(
            self.origin, ProposalOrigin
        ):
            raise ProofStrategyBoundaryError("proposal enum value is unregistered")
        _hash(self.parameters_sha256, "parameters hash")
        if self.origin is ProposalOrigin.LLM and self.source is None:
            raise ProofStrategyBoundaryError("LLM proposal requires a source binding")
        _hash(self.proposal_sha256, "proposal hash")
        expected = canonical_sha256(self._body())
        if self.proposal_sha256 != expected:
            raise ProofStrategyBoundaryError("proposal canonical hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "artifact": self.artifact.to_dict(),
            "strategy": self.strategy.value,
            "origin": self.origin.value,
            "parameters_sha256": self.parameters_sha256,
            "source": None if self.source is None else self.source.to_dict(),
        }

    @classmethod
    def create(
        cls,
        proposal_id: str,
        artifact: ArtifactRef,
        strategy: ProofStrategy,
        origin: ProposalOrigin,
        parameters: dict[str, Any],
        *,
        source: SourceBinding | None = None,
    ) -> ProofStrategyProposal:
        body = {
            "schema_version": "sigma-math-proof-strategy-proposal-1.0",
            "proposal_id": _identifier(proposal_id, "proposal ID"),
            "artifact": artifact.to_dict(),
            "strategy": strategy.value,
            "origin": origin.value,
            "parameters_sha256": canonical_sha256(parameters),
            "source": None if source is None else source.to_dict(),
        }
        return cls(
            proposal_id,
            artifact,
            strategy,
            origin,
            body["parameters_sha256"],
            source,
            canonical_sha256(body),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "proposal_sha256": self.proposal_sha256}


@dataclass(frozen=True, slots=True)
class ProofAttempt:
    proposal_sha256: str
    artifact: ArtifactRef
    status: ProofAttemptStatus
    executor_id: str
    certificate_sha256: str | None
    reason_codes: tuple[str, ...]
    attempt_sha256: str
    schema_version: str = "sigma-math-proof-attempt-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-math-proof-attempt-1.0":
            raise ProofStrategyBoundaryError("proof attempt schema changed")
        _hash(self.proposal_sha256, "proposal hash")
        _identifier(self.executor_id, "executor ID")
        if not isinstance(self.status, ProofAttemptStatus):
            raise ProofStrategyBoundaryError("proof attempt status is unregistered")
        reasons = tuple(_identifier(reason, "reason code") for reason in self.reason_codes)
        if reasons != tuple(sorted(set(reasons))):
            raise ProofStrategyBoundaryError("reason codes must be sorted and unique")
        if self.status is ProofAttemptStatus.PROVED:
            if self.certificate_sha256 is None or reasons:
                raise ProofStrategyBoundaryError(
                    "proved attempt requires a certificate and no reason"
                )
            _hash(self.certificate_sha256, "certificate hash")
        elif self.certificate_sha256 is not None or not reasons:
            raise ProofStrategyBoundaryError(
                "non-proved attempt requires reasons and no certificate"
            )
        _hash(self.attempt_sha256, "attempt hash")
        if self.attempt_sha256 != canonical_sha256(self._body()):
            raise ProofStrategyBoundaryError("proof attempt canonical hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal_sha256": self.proposal_sha256,
            "artifact": self.artifact.to_dict(),
            "status": self.status.value,
            "executor_id": self.executor_id,
            "certificate_sha256": self.certificate_sha256,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def create(
        cls,
        proposal: ProofStrategyProposal,
        status: ProofAttemptStatus,
        executor_id: str,
        *,
        certificate_sha256: str | None = None,
        reason_codes: tuple[str, ...] = (),
    ) -> ProofAttempt:
        body = {
            "schema_version": "sigma-math-proof-attempt-1.0",
            "proposal_sha256": proposal.proposal_sha256,
            "artifact": proposal.artifact.to_dict(),
            "status": status.value,
            "executor_id": executor_id,
            "certificate_sha256": certificate_sha256,
            "reason_codes": sorted(reason_codes),
        }
        return cls(
            proposal.proposal_sha256,
            proposal.artifact,
            status,
            executor_id,
            certificate_sha256,
            tuple(body["reason_codes"]),
            canonical_sha256(body),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "attempt_sha256": self.attempt_sha256}


def deterministic_strategy_proposals(
    artifact: ArtifactRef,
    statement: Equation | Recurrence,
) -> tuple[ProofStrategyProposal, ...]:
    """Return strategy suggestions only; suggestions never imply validity."""

    strategies = (
        (ProofStrategy.EXACT_ALGEBRA, {"reason": "equation_in_exact_rational_ir"}),
        (ProofStrategy.FACTORIZATION, {"reason": "exact_algebra_secondary_strategy"}),
    )
    if isinstance(statement, Recurrence):
        strategies = (
            (ProofStrategy.INDUCTION, {"reason": "registered_forward_recurrence"}),
            *strategies,
        )
    return tuple(
        ProofStrategyProposal.create(
            f"deterministic-{index:02d}-{strategy.value}",
            artifact,
            strategy,
            ProposalOrigin.DETERMINISTIC,
            parameters,
        )
        for index, (strategy, parameters) in enumerate(strategies)
    )


def execute_strategy(
    proposal: ProofStrategyProposal,
    statement: Equation,
    *,
    recurrence: Recurrence | None = None,
    base_index: int | None = None,
) -> ProofAttempt:
    """Execute a registered deterministic strategy or return a typed block/error."""

    if proposal.strategy is ProofStrategy.EXACT_ALGEBRA:
        if not isinstance(statement, Equation):
            return ProofAttempt.create(
                proposal,
                ProofAttemptStatus.BLOCKED,
                "sympy-rational-identity-v1",
                reason_codes=("strategy_statement_kind_mismatch",),
            )
        try:
            certificate = prove_rational_identity(statement)
        except ProofFailure:
            return ProofAttempt.create(
                proposal,
                ProofAttemptStatus.REFUTED,
                "sympy-rational-identity-v1",
                reason_codes=("exact_identity_failed",),
            )
        except UnsupportedProof:
            return ProofAttempt.create(
                proposal,
                ProofAttemptStatus.BLOCKED,
                "sympy-rational-identity-v1",
                reason_codes=("exact_identity_unsupported",),
            )
        return ProofAttempt.create(
            proposal,
            ProofAttemptStatus.PROVED,
            "sympy-rational-identity-v1",
            certificate_sha256=certificate["content_sha256"],
        )
    if proposal.strategy is ProofStrategy.INDUCTION:
        if recurrence is None or base_index is None:
            return ProofAttempt.create(
                proposal,
                ProofAttemptStatus.BLOCKED,
                "sympy-first-order-induction-v1",
                reason_codes=("induction_inputs_incomplete",),
            )
        try:
            certificate = prove_induction(statement, recurrence, base_index=base_index)
        except ProofFailure:
            return ProofAttempt.create(
                proposal,
                ProofAttemptStatus.REFUTED,
                "sympy-first-order-induction-v1",
                reason_codes=("induction_failed",),
            )
        except UnsupportedProof:
            return ProofAttempt.create(
                proposal,
                ProofAttemptStatus.BLOCKED,
                "sympy-first-order-induction-v1",
                reason_codes=("induction_unsupported",),
            )
        return ProofAttempt.create(
            proposal,
            ProofAttemptStatus.PROVED,
            "sympy-first-order-induction-v1",
            certificate_sha256=certificate["content_sha256"],
        )
    return ProofAttempt.create(
        proposal,
        ProofAttemptStatus.BLOCKED,
        "unregistered-strategy-executor-v1",
        reason_codes=("strategy_executor_not_registered",),
    )


__all__ = [
    "ProofAttempt",
    "ProofAttemptStatus",
    "ProofStrategy",
    "ProofStrategyBoundaryError",
    "ProofStrategyProposal",
    "ProposalOrigin",
    "deterministic_strategy_proposals",
    "execute_strategy",
]
