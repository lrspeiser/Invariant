"""Exact, bounded Bayesian proposal prioritization for Sigma candidate artifacts.

This module deliberately treats posterior mass as a proposal-ordering device only.  A posterior
does not establish truth, scientific validity, gate passage, or promotion eligibility.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from sigma_theory_compiler.sigma_core import (
    CandidateArtifact,
    SchemaViolation,
    canonical_json_bytes,
    canonical_sha256,
)

STATE_SCHEMA = "sigma-bayesian-candidate-state-1.0"
EVIDENCE_SCHEMA = "sigma-bayesian-evidence-1.0"
PROPOSAL_SCHEMA = "sigma-bayesian-proposal-batch-1.0"
SCOPE = (
    "Exact posterior mass prioritizes bounded proposals only; it does not imply truth, "
    "scientific validity, gate passage, or promotion eligibility."
)
_MAX_CANDIDATES = 10_000
_MAX_EVIDENCE_UPDATES = 10_000
_MAX_PROPOSAL_DRAWS = 100_000
_MAX_INTEGER_BITS = 4_096
_MAX_REJECTION_ATTEMPTS = 1_024
_SHA256_LENGTH = 64


class BayesianGeneratorError(ValueError):
    """A Bayesian input or transition violates the fail-closed boundary."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise BayesianGeneratorError(f"{label} keys changed")


def _positive_bounded_integer(value: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise BayesianGeneratorError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BayesianGeneratorError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ExactProbability:
    """A canonical exact probability in ``[0, 1]``."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.numerator, bool)
            or isinstance(self.denominator, bool)
            or not isinstance(self.numerator, int)
            or not isinstance(self.denominator, int)
        ):
            raise BayesianGeneratorError("probability components must be integers")
        if self.denominator <= 0 or not 0 <= self.numerator <= self.denominator:
            raise BayesianGeneratorError("probability must lie in [0, 1]")
        if math.gcd(self.numerator, self.denominator) != 1:
            raise BayesianGeneratorError("probability must be in lowest terms")
        if max(self.numerator.bit_length(), self.denominator.bit_length()) > _MAX_INTEGER_BITS:
            raise BayesianGeneratorError("probability exceeds the exact-integer bit budget")

    @classmethod
    def from_fraction(cls, value: Fraction) -> ExactProbability:
        if not isinstance(value, Fraction):
            raise BayesianGeneratorError("probability source must be fractions.Fraction")
        return cls(value.numerator, value.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExactProbability:
        _exact_keys(value, {"numerator", "denominator"}, "exact probability")
        return cls(value["numerator"], value["denominator"])


@dataclass(frozen=True, slots=True)
class WeightedCandidate:
    """A typed Sigma candidate and its positive, not-necessarily-normalized prior."""

    artifact: CandidateArtifact
    prior: ExactProbability

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, CandidateArtifact):
            raise BayesianGeneratorError("weighted candidate must contain a CandidateArtifact")
        try:
            self.artifact.validate()
        except SchemaViolation as error:
            raise BayesianGeneratorError(
                "candidate artifact failed canonical validation"
            ) from error
        if self.prior.numerator == 0:
            raise BayesianGeneratorError("candidate prior must be positive")


@dataclass(frozen=True, slots=True)
class CandidatePosterior:
    """A canonical candidate with normalized prior and current posterior."""

    artifact: CandidateArtifact
    prior: ExactProbability
    posterior: ExactProbability

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, CandidateArtifact):
            raise BayesianGeneratorError("posterior candidate must contain a CandidateArtifact")
        try:
            self.artifact.validate()
        except SchemaViolation as error:
            raise BayesianGeneratorError(
                "candidate artifact failed canonical validation"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "prior": self.prior.to_dict(),
            "posterior": self.posterior.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidatePosterior:
        _exact_keys(value, {"artifact", "prior", "posterior"}, "candidate posterior")
        try:
            artifact = CandidateArtifact.from_dict(value["artifact"])
        except (SchemaViolation, TypeError, ValueError) as error:
            raise BayesianGeneratorError(
                "candidate artifact failed canonical validation"
            ) from error
        return cls(
            artifact=artifact,
            prior=ExactProbability.from_dict(value["prior"]),
            posterior=ExactProbability.from_dict(value["posterior"]),
        )


@dataclass(frozen=True, slots=True)
class EvidenceLikelihood:
    artifact_id: str
    likelihood: ExactProbability

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.startswith("sig-"):
            raise BayesianGeneratorError("evidence artifact_id is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "likelihood": self.likelihood.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceLikelihood:
        _exact_keys(value, {"artifact_id", "likelihood"}, "evidence likelihood")
        return cls(str(value["artifact_id"]), ExactProbability.from_dict(value["likelihood"]))


@dataclass(frozen=True, slots=True)
class EvidenceBatch:
    """A closed exact likelihood vector for one evidence observation."""

    evidence_id: str
    likelihoods: tuple[EvidenceLikelihood, ...]
    content_sha256: str
    schema_version: str = EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA:
            raise BayesianGeneratorError("evidence schema_version changed")
        if (
            not isinstance(self.evidence_id, str)
            or not self.evidence_id
            or self.evidence_id != self.evidence_id.strip()
        ):
            raise BayesianGeneratorError("evidence_id must be nonempty and stripped")
        if not self.likelihoods:
            raise BayesianGeneratorError("evidence likelihood vector must be nonempty")
        ids = tuple(item.artifact_id for item in self.likelihoods)
        if ids != tuple(sorted(set(ids))):
            raise BayesianGeneratorError("evidence likelihoods must be unique and sorted")
        _sha256(self.content_sha256, "evidence content_sha256")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise BayesianGeneratorError("evidence canonical hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "likelihoods": [item.to_dict() for item in self.likelihoods],
        }

    @classmethod
    def create(cls, evidence_id: str, likelihoods: Sequence[EvidenceLikelihood]) -> EvidenceBatch:
        ordered = tuple(sorted(likelihoods, key=lambda item: item.artifact_id))
        body = {
            "schema_version": EVIDENCE_SCHEMA,
            "evidence_id": evidence_id,
            "likelihoods": [item.to_dict() for item in ordered],
        }
        return cls(evidence_id, ordered, canonical_sha256(body))

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceBatch:
        _exact_keys(
            value,
            {"schema_version", "evidence_id", "likelihoods", "content_sha256"},
            "evidence batch",
        )
        if not isinstance(value["likelihoods"], list):
            raise BayesianGeneratorError("evidence likelihoods must be an array")
        return cls(
            evidence_id=str(value["evidence_id"]),
            likelihoods=tuple(EvidenceLikelihood.from_dict(item) for item in value["likelihoods"]),
            content_sha256=str(value["content_sha256"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class BayesianBudget:
    max_candidates: int
    max_evidence_updates: int
    max_proposal_draws: int

    def __post_init__(self) -> None:
        _positive_bounded_integer(self.max_candidates, _MAX_CANDIDATES, "max_candidates")
        _positive_bounded_integer(
            self.max_evidence_updates, _MAX_EVIDENCE_UPDATES, "max_evidence_updates"
        )
        _positive_bounded_integer(
            self.max_proposal_draws, _MAX_PROPOSAL_DRAWS, "max_proposal_draws"
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_candidates": self.max_candidates,
            "max_evidence_updates": self.max_evidence_updates,
            "max_proposal_draws": self.max_proposal_draws,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BayesianBudget:
        _exact_keys(
            value,
            {"max_candidates", "max_evidence_updates", "max_proposal_draws"},
            "Bayesian budget",
        )
        return cls(
            value["max_candidates"], value["max_evidence_updates"], value["max_proposal_draws"]
        )


def _normalize(values: Sequence[Fraction]) -> tuple[ExactProbability, ...]:
    total = sum(values, start=Fraction(0, 1))
    if total <= 0:
        raise BayesianGeneratorError("probability mass is zero")
    return tuple(ExactProbability.from_fraction(value / total) for value in values)


@dataclass(frozen=True, slots=True)
class BayesianState:
    """A sealed prior/posterior state with append-only evidence lineage."""

    candidates: tuple[CandidatePosterior, ...]
    evidence_history: tuple[EvidenceBatch, ...]
    budget: BayesianBudget
    deduplicated_input_count: int
    parent_state_sha256: str | None
    lineage_sha256: str
    content_sha256: str
    scope: str = SCOPE
    schema_version: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATE_SCHEMA or self.scope != SCOPE:
            raise BayesianGeneratorError("Bayesian state schema or scope changed")
        if not self.candidates or len(self.candidates) > self.budget.max_candidates:
            raise BayesianGeneratorError("Bayesian state candidate budget violated")
        if len(self.evidence_history) > self.budget.max_evidence_updates:
            raise BayesianGeneratorError("Bayesian state evidence budget violated")
        if (
            isinstance(self.deduplicated_input_count, bool)
            or not isinstance(self.deduplicated_input_count, int)
            or self.deduplicated_input_count < 0
        ):
            raise BayesianGeneratorError("deduplicated_input_count must be nonnegative")
        ids = tuple(item.artifact.artifact_id for item in self.candidates)
        if ids != tuple(sorted(set(ids))):
            raise BayesianGeneratorError("state candidates must be unique and sorted")
        if sum((item.prior.fraction for item in self.candidates), Fraction()) != 1:
            raise BayesianGeneratorError("normalized priors must sum to one")
        if sum((item.posterior.fraction for item in self.candidates), Fraction()) != 1:
            raise BayesianGeneratorError("posteriors must sum to one")
        evidence_ids = tuple(item.evidence_id for item in self.evidence_history)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise BayesianGeneratorError("evidence history IDs contain duplicates")
        if self.parent_state_sha256 is not None:
            _sha256(self.parent_state_sha256, "parent_state_sha256")
        _sha256(self.lineage_sha256, "lineage_sha256")
        _sha256(self.content_sha256, "state content_sha256")
        if self.lineage_sha256 != canonical_sha256(self._lineage_body()):
            raise BayesianGeneratorError("state lineage hash changed")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise BayesianGeneratorError("state canonical hash changed")

    def _lineage_body(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA,
            "parent_state_sha256": self.parent_state_sha256,
            "candidate_provenance": [
                {
                    "artifact_id": item.artifact.artifact_id,
                    "content_sha256": item.artifact.content_sha256,
                    "provenance_sha256": canonical_sha256(item.artifact.provenance.to_dict()),
                }
                for item in self.candidates
            ],
            "evidence_receipts": [item.content_sha256 for item in self.evidence_history],
        }

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "budget": self.budget.to_dict(),
            "deduplicated_input_count": self.deduplicated_input_count,
            "parent_state_sha256": self.parent_state_sha256,
            "lineage_sha256": self.lineage_sha256,
            "candidates": [item.to_dict() for item in self.candidates],
            "evidence_history": [item.to_dict() for item in self.evidence_history],
        }

    @classmethod
    def create(
        cls, weighted_candidates: Sequence[WeightedCandidate], budget: BayesianBudget
    ) -> BayesianState:
        if not weighted_candidates:
            raise BayesianGeneratorError("at least one weighted candidate is required")
        combined: dict[str, tuple[CandidateArtifact, Fraction]] = {}
        for item in weighted_candidates:
            if not isinstance(item, WeightedCandidate):
                raise BayesianGeneratorError("candidate inputs must be WeightedCandidate values")
            digest = item.artifact.content_sha256
            if digest in combined:
                artifact, mass = combined[digest]
                if artifact.artifact_id != item.artifact.artifact_id:
                    raise BayesianGeneratorError("canonical candidate identity collision")
                combined[digest] = (artifact, mass + item.prior.fraction)
            else:
                combined[digest] = (item.artifact, item.prior.fraction)
        if len(combined) > budget.max_candidates:
            raise BayesianGeneratorError("input candidates exceed max_candidates")
        ordered = sorted(combined.values(), key=lambda item: item[0].artifact_id)
        normalized = _normalize([mass for _, mass in ordered])
        candidates = tuple(
            CandidatePosterior(artifact, probability, probability)
            for (artifact, _), probability in zip(ordered, normalized, strict=True)
        )
        return cls._seal(
            candidates=candidates,
            evidence_history=(),
            budget=budget,
            deduplicated_input_count=len(weighted_candidates) - len(candidates),
            parent_state_sha256=None,
        )

    @classmethod
    def _seal(
        cls,
        *,
        candidates: tuple[CandidatePosterior, ...],
        evidence_history: tuple[EvidenceBatch, ...],
        budget: BayesianBudget,
        deduplicated_input_count: int,
        parent_state_sha256: str | None,
    ) -> BayesianState:
        provisional = object.__new__(cls)
        for name, value in {
            "candidates": candidates,
            "evidence_history": evidence_history,
            "budget": budget,
            "deduplicated_input_count": deduplicated_input_count,
            "parent_state_sha256": parent_state_sha256,
            "scope": SCOPE,
            "schema_version": STATE_SCHEMA,
        }.items():
            object.__setattr__(provisional, name, value)
        lineage = canonical_sha256(provisional._lineage_body())
        object.__setattr__(provisional, "lineage_sha256", lineage)
        content = canonical_sha256(provisional._body())
        return cls(
            candidates=candidates,
            evidence_history=evidence_history,
            budget=budget,
            deduplicated_input_count=deduplicated_input_count,
            parent_state_sha256=parent_state_sha256,
            lineage_sha256=lineage,
            content_sha256=content,
        )

    def update(self, evidence: EvidenceBatch) -> BayesianState:
        if not isinstance(evidence, EvidenceBatch):
            raise BayesianGeneratorError("update requires a sealed EvidenceBatch")
        if len(self.evidence_history) >= self.budget.max_evidence_updates:
            raise BayesianGeneratorError("max_evidence_updates exhausted")
        if evidence.evidence_id in {item.evidence_id for item in self.evidence_history}:
            raise BayesianGeneratorError("evidence_id replay is forbidden")
        likelihoods = {item.artifact_id: item.likelihood.fraction for item in evidence.likelihoods}
        candidate_ids = {item.artifact.artifact_id for item in self.candidates}
        if set(likelihoods) != candidate_ids:
            raise BayesianGeneratorError("evidence must cover exactly the candidate state")
        products = [
            item.posterior.fraction * likelihoods[item.artifact.artifact_id]
            for item in self.candidates
        ]
        normalized = _normalize(products)
        updated = tuple(
            CandidatePosterior(item.artifact, item.prior, posterior)
            for item, posterior in zip(self.candidates, normalized, strict=True)
        )
        return self._seal(
            candidates=updated,
            evidence_history=(*self.evidence_history, evidence),
            budget=self.budget,
            deduplicated_input_count=self.deduplicated_input_count,
            parent_state_sha256=self.content_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BayesianState:
        _exact_keys(
            value,
            {
                "schema_version",
                "scope",
                "budget",
                "deduplicated_input_count",
                "parent_state_sha256",
                "lineage_sha256",
                "candidates",
                "evidence_history",
                "content_sha256",
            },
            "Bayesian state",
        )
        if not isinstance(value["candidates"], list) or not isinstance(
            value["evidence_history"], list
        ):
            raise BayesianGeneratorError("state candidates and evidence_history must be arrays")
        return cls(
            candidates=tuple(CandidatePosterior.from_dict(item) for item in value["candidates"]),
            evidence_history=tuple(
                EvidenceBatch.from_dict(item) for item in value["evidence_history"]
            ),
            budget=BayesianBudget.from_dict(value["budget"]),
            deduplicated_input_count=value["deduplicated_input_count"],
            parent_state_sha256=value["parent_state_sha256"],
            lineage_sha256=str(value["lineage_sha256"]),
            content_sha256=str(value["content_sha256"]),
            scope=str(value["scope"]),
            schema_version=str(value["schema_version"]),
        )


def _integer_ticket_weights(state: BayesianState) -> tuple[int, ...]:
    denominator = 1
    for item in state.candidates:
        denominator = math.lcm(denominator, item.posterior.denominator)
        if denominator.bit_length() > _MAX_INTEGER_BITS:
            raise BayesianGeneratorError("posterior ticket denominator exceeds bit budget")
    weights = tuple(
        item.posterior.numerator * (denominator // item.posterior.denominator)
        for item in state.candidates
    )
    if sum(weights).bit_length() > _MAX_INTEGER_BITS:
        raise BayesianGeneratorError("posterior ticket total exceeds bit budget")
    return weights


def _deterministic_below(limit: int, material: bytes, draw: int) -> int:
    bit_count = limit.bit_length()
    byte_count = (bit_count + 7) // 8
    mask = (1 << bit_count) - 1
    for attempt in range(_MAX_REJECTION_ATTEMPTS):
        blocks = bytearray()
        block = 0
        while len(blocks) < byte_count:
            blocks.extend(
                hashlib.sha256(
                    material
                    + draw.to_bytes(8, "big")
                    + attempt.to_bytes(4, "big")
                    + block.to_bytes(4, "big")
                ).digest()
            )
            block += 1
        value = int.from_bytes(blocks[:byte_count], "big") & mask
        if value < limit:
            return value
    raise BayesianGeneratorError("deterministic exact sampler exhausted rejection budget")


@dataclass(frozen=True, slots=True)
class BayesianProposalBatch:
    seed: int
    requested_draws: int
    draw_artifact_ids: tuple[str, ...]
    proposals: tuple[CandidateArtifact, ...]
    duplicates_removed: int
    source_state_sha256: str
    lineage_sha256: str
    content_sha256: str
    scope: str = SCOPE
    schema_version: str = PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROPOSAL_SCHEMA or self.scope != SCOPE:
            raise BayesianGeneratorError("proposal schema or scope changed")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**64
        ):
            raise BayesianGeneratorError("seed must be an unsigned 64-bit integer")
        _positive_bounded_integer(self.requested_draws, _MAX_PROPOSAL_DRAWS, "requested_draws")
        if self.requested_draws != len(self.draw_artifact_ids):
            raise BayesianGeneratorError("proposal draw count changed")
        if self.duplicates_removed != self.requested_draws - len(self.proposals):
            raise BayesianGeneratorError("proposal duplicate count changed")
        _sha256(self.source_state_sha256, "source_state_sha256")
        _sha256(self.lineage_sha256, "proposal lineage_sha256")
        _sha256(self.content_sha256, "proposal content_sha256")
        for proposal in self.proposals:
            try:
                proposal.validate()
            except SchemaViolation as error:
                raise BayesianGeneratorError("proposal candidate failed validation") from error
        by_id = {item.artifact_id: item for item in self.proposals}
        if len(by_id) != len(self.proposals) or set(self.draw_artifact_ids) != set(by_id):
            raise BayesianGeneratorError("proposal canonical dedup boundary changed")
        first_seen = tuple(dict.fromkeys(self.draw_artifact_ids))
        if tuple(item.artifact_id for item in self.proposals) != first_seen:
            raise BayesianGeneratorError("proposals must follow first-draw order")
        if self.lineage_sha256 != canonical_sha256(self._lineage_body()):
            raise BayesianGeneratorError("proposal lineage hash changed")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise BayesianGeneratorError("proposal canonical hash changed")

    def _lineage_body(self) -> dict[str, Any]:
        return {
            "schema_version": PROPOSAL_SCHEMA,
            "source_state_sha256": self.source_state_sha256,
            "seed": self.seed,
            "draw_artifact_ids": list(self.draw_artifact_ids),
        }

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "seed": self.seed,
            "requested_draws": self.requested_draws,
            "draw_artifact_ids": list(self.draw_artifact_ids),
            "proposals": [item.to_dict() for item in self.proposals],
            "duplicates_removed": self.duplicates_removed,
            "source_state_sha256": self.source_state_sha256,
            "lineage_sha256": self.lineage_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BayesianProposalBatch:
        _exact_keys(
            value,
            {
                "schema_version",
                "scope",
                "seed",
                "requested_draws",
                "draw_artifact_ids",
                "proposals",
                "duplicates_removed",
                "source_state_sha256",
                "lineage_sha256",
                "content_sha256",
            },
            "Bayesian proposal batch",
        )
        if not isinstance(value["draw_artifact_ids"], list) or not isinstance(
            value["proposals"], list
        ):
            raise BayesianGeneratorError("proposal draws and proposals must be arrays")
        try:
            proposals = tuple(CandidateArtifact.from_dict(item) for item in value["proposals"])
        except (SchemaViolation, TypeError, ValueError) as error:
            raise BayesianGeneratorError("proposal candidate failed validation") from error
        return cls(
            seed=value["seed"],
            requested_draws=value["requested_draws"],
            draw_artifact_ids=tuple(value["draw_artifact_ids"]),
            proposals=proposals,
            duplicates_removed=value["duplicates_removed"],
            source_state_sha256=str(value["source_state_sha256"]),
            lineage_sha256=str(value["lineage_sha256"]),
            content_sha256=str(value["content_sha256"]),
            scope=str(value["scope"]),
            schema_version=str(value["schema_version"]),
        )


class BayesianCandidateGenerator:
    """Deterministically sample a bounded proposal batch from exact posterior mass."""

    @staticmethod
    def propose(state: BayesianState, *, seed: int, draws: int) -> BayesianProposalBatch:
        if not isinstance(state, BayesianState):
            raise BayesianGeneratorError("propose requires a sealed BayesianState")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise BayesianGeneratorError("seed must be an unsigned 64-bit integer")
        _positive_bounded_integer(draws, state.budget.max_proposal_draws, "draws")
        weights = _integer_ticket_weights(state)
        total = sum(weights)
        cumulative: list[int] = []
        running = 0
        for weight in weights:
            running += weight
            cumulative.append(running)
        material = canonical_json_bytes(
            {
                "schema_version": PROPOSAL_SCHEMA,
                "source_state_sha256": state.content_sha256,
                "seed": seed,
            }
        )
        draw_ids = []
        for draw in range(draws):
            ticket = _deterministic_below(total, material, draw)
            index = next(i for i, upper in enumerate(cumulative) if ticket < upper)
            draw_ids.append(state.candidates[index].artifact.artifact_id)
        by_id = {item.artifact.artifact_id: item.artifact for item in state.candidates}
        proposals = tuple(by_id[artifact_id] for artifact_id in dict.fromkeys(draw_ids))
        provisional = object.__new__(BayesianProposalBatch)
        fields = {
            "seed": seed,
            "requested_draws": draws,
            "draw_artifact_ids": tuple(draw_ids),
            "proposals": proposals,
            "duplicates_removed": draws - len(proposals),
            "source_state_sha256": state.content_sha256,
            "scope": SCOPE,
            "schema_version": PROPOSAL_SCHEMA,
        }
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        lineage = canonical_sha256(provisional._lineage_body())
        object.__setattr__(provisional, "lineage_sha256", lineage)
        content = canonical_sha256(provisional._body())
        return BayesianProposalBatch(
            **fields,
            lineage_sha256=lineage,
            content_sha256=content,
        )
