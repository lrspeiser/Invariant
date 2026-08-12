"""Deterministic, bounded evolution of Sigma Core candidate artifacts.

This module generates candidates and records heuristic evaluation outcomes.  It deliberately has
no promotion API and makes no scientific-truth claim: a ``pass`` means only that a caller-supplied
bounded evaluator made the candidate eligible for the next generation.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .sigma_core import (
    ArtifactRef,
    CandidateArtifact,
    DomainPackRef,
    OutcomeStatus,
    SchemaViolation,
    canonical_sha256,
)

SCHEMA_VERSION = "sigma-evolutionary-candidate-generator-1.0"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SchemaViolation(f"{label} keys changed")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} is not a canonical identifier")
    return value


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaViolation(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaViolation(f"{label} must be an integer >= {minimum}")
    return value


def _reason_codes(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_identifier(item, "reason code") for item in values)
    if not result or result != tuple(sorted(set(result))):
        raise SchemaViolation("reason codes must be nonempty, unique, and sorted")
    return result


@dataclass(frozen=True, slots=True)
class EvolutionBudget:
    """Closed resource boundary for one evolutionary run."""

    population_size: int
    generations: int
    offspring_per_generation: int
    max_evaluations: int

    def __post_init__(self) -> None:
        _strict_int(self.population_size, "population_size", minimum=1)
        _strict_int(self.generations, "generations", minimum=0)
        _strict_int(self.offspring_per_generation, "offspring_per_generation", minimum=0)
        _strict_int(self.max_evaluations, "max_evaluations", minimum=1)

    @property
    def max_operator_attempts(self) -> int:
        return self.generations * self.offspring_per_generation

    def to_dict(self) -> dict[str, int]:
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "offspring_per_generation": self.offspring_per_generation,
            "max_evaluations": self.max_evaluations,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvolutionBudget:
        _exact_keys(
            value,
            {"population_size", "generations", "offspring_per_generation", "max_evaluations"},
            "evolution budget",
        )
        return cls(
            value["population_size"],
            value["generations"],
            value["offspring_per_generation"],
            value["max_evaluations"],
        )


class SeedStream:
    """Small SHA-256 counter stream with deterministic bounded selection."""

    def __init__(self, seed: str) -> None:
        if not isinstance(seed, str) or not seed or seed != seed.strip():
            raise SchemaViolation("evolution seed must be a nonempty stripped string")
        self._seed = seed.encode("utf-8")
        self._counter = 0

    @property
    def draws(self) -> int:
        return self._counter

    def draw(self, upper_bound: int) -> int:
        _strict_int(upper_bound, "draw upper_bound", minimum=1)
        block = hashlib.sha256(
            self._seed + b"\0" + self._counter.to_bytes(16, "big", signed=False)
        ).digest()
        self._counter += 1
        return int.from_bytes(block, "big") % upper_bound

    def choose(self, values: Sequence[CandidateArtifact]) -> CandidateArtifact:
        if not values:
            raise SchemaViolation("cannot choose from an empty candidate sequence")
        return values[self.draw(len(values))]


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Typed heuristic result; ``pass`` is selection eligibility, never promotion."""

    artifact: ArtifactRef
    status: OutcomeStatus
    score: int | None
    reason_codes: tuple[str, ...]
    outcome_sha256: str
    schema_version: str = "sigma-evolution-evaluation-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-evolution-evaluation-1.0":
            raise SchemaViolation("evaluation schema_version changed")
        if self.status is OutcomeStatus.PASS:
            if isinstance(self.score, bool) or not isinstance(self.score, int):
                raise SchemaViolation("pass score must be an integer")
            if self.reason_codes:
                raise SchemaViolation("pass evaluation cannot carry reason codes")
        else:
            if self.score is not None:
                raise SchemaViolation("non-pass evaluation cannot carry a score")
            _reason_codes(self.reason_codes)
        _digest(self.outcome_sha256, "evaluation outcome_sha256")
        if self.outcome_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("evaluation canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact": self.artifact.to_dict(),
            "status": self.status.value,
            "score": self.score,
            "reason_codes": list(self.reason_codes),
            "selection_only": True,
            "physics_truth_established": False,
            "promotion_allowed": False,
        }

    @classmethod
    def create(
        cls,
        artifact: CandidateArtifact | ArtifactRef,
        status: OutcomeStatus,
        *,
        score: int | None = None,
        reason_codes: Sequence[str] = (),
    ) -> EvaluationOutcome:
        ref = artifact.ref if isinstance(artifact, CandidateArtifact) else artifact
        reasons = tuple(sorted(reason_codes))
        body = {
            "schema_version": "sigma-evolution-evaluation-1.0",
            "artifact": ref.to_dict(),
            "status": status.value,
            "score": score,
            "reason_codes": list(reasons),
            "selection_only": True,
            "physics_truth_established": False,
            "promotion_allowed": False,
        }
        return cls(ref, status, score, reasons, canonical_sha256(body))

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "outcome_sha256": self.outcome_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationOutcome:
        _exact_keys(
            value,
            {
                "schema_version",
                "artifact",
                "status",
                "score",
                "reason_codes",
                "selection_only",
                "physics_truth_established",
                "promotion_allowed",
                "outcome_sha256",
            },
            "evaluation outcome",
        )
        if (
            value["selection_only"] is not True
            or value["physics_truth_established"] is not False
            or value["promotion_allowed"] is not False
            or not isinstance(value["reason_codes"], list)
        ):
            raise SchemaViolation("evaluation scientific boundary changed")
        try:
            status = OutcomeStatus(value["status"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("evaluation status is not registered") from error
        return cls(
            ArtifactRef.from_dict(value["artifact"]),
            status,
            value["score"],
            tuple(value["reason_codes"]),
            str(value["outcome_sha256"]),
            str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class LineageRecord:
    generation: int
    ordinal: int
    operation: str
    child: ArtifactRef
    parents: tuple[ArtifactRef, ...]
    lineage_sha256: str
    schema_version: str = "sigma-evolution-lineage-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-evolution-lineage-1.0":
            raise SchemaViolation("lineage schema_version changed")
        _strict_int(self.generation, "lineage generation")
        _strict_int(self.ordinal, "lineage ordinal")
        if self.operation not in {"seed", "mutation", "crossover"}:
            raise SchemaViolation("lineage operation is not registered")
        if (self.operation == "seed") != (self.generation == 0):
            raise SchemaViolation("seed lineage must be generation zero only")
        expected_parent_count = {"seed": 0, "mutation": 1, "crossover": 2}[self.operation]
        if len(self.parents) != expected_parent_count:
            raise SchemaViolation("lineage parent count changed")
        if self.parents != tuple(sorted(self.parents, key=lambda item: item.artifact_id)):
            raise SchemaViolation("lineage parents must be sorted")
        if len({item.artifact_id for item in self.parents}) != len(self.parents):
            raise SchemaViolation("lineage parents contain duplicates")
        _digest(self.lineage_sha256, "lineage_sha256")
        if self.lineage_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("lineage canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "ordinal": self.ordinal,
            "operation": self.operation,
            "child": self.child.to_dict(),
            "parents": [item.to_dict() for item in self.parents],
        }

    @classmethod
    def create(
        cls,
        generation: int,
        ordinal: int,
        operation: str,
        child: CandidateArtifact,
        parents: Sequence[CandidateArtifact],
    ) -> LineageRecord:
        refs = tuple(sorted((item.ref for item in parents), key=lambda item: item.artifact_id))
        body = {
            "schema_version": "sigma-evolution-lineage-1.0",
            "generation": generation,
            "ordinal": ordinal,
            "operation": operation,
            "child": child.ref.to_dict(),
            "parents": [item.to_dict() for item in refs],
        }
        return cls(generation, ordinal, operation, child.ref, refs, canonical_sha256(body))

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "lineage_sha256": self.lineage_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LineageRecord:
        _exact_keys(
            value,
            {
                "schema_version",
                "generation",
                "ordinal",
                "operation",
                "child",
                "parents",
                "lineage_sha256",
            },
            "lineage record",
        )
        if not isinstance(value["parents"], list):
            raise SchemaViolation("lineage parents must be an array")
        return cls(
            value["generation"],
            value["ordinal"],
            str(value["operation"]),
            ArtifactRef.from_dict(value["child"]),
            tuple(ArtifactRef.from_dict(item) for item in value["parents"]),
            str(value["lineage_sha256"]),
            str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class EvolutionEvent:
    generation: int
    ordinal: int
    operation: str
    status: OutcomeStatus
    parents: tuple[ArtifactRef, ...]
    child: ArtifactRef | None
    reason_codes: tuple[str, ...]
    event_sha256: str
    schema_version: str = "sigma-evolution-event-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-evolution-event-1.0":
            raise SchemaViolation("event schema_version changed")
        _strict_int(self.generation, "event generation")
        _strict_int(self.ordinal, "event ordinal")
        if self.operation not in {"seed", "mutation", "crossover"}:
            raise SchemaViolation("event operation is not registered")
        expected_parent_count = {"seed": 0, "mutation": 1, "crossover": 2}[self.operation]
        if len(self.parents) != expected_parent_count:
            raise SchemaViolation("event parent count changed")
        if self.parents != tuple(sorted(self.parents, key=lambda item: item.artifact_id)):
            raise SchemaViolation("event parents must be sorted")
        if len({item.artifact_id for item in self.parents}) != len(self.parents):
            raise SchemaViolation("event parents contain duplicates")
        if self.status is OutcomeStatus.PASS:
            if self.reason_codes or self.child is None:
                raise SchemaViolation("pass event must have a child and no reasons")
        else:
            _reason_codes(self.reason_codes)
        _digest(self.event_sha256, "event_sha256")
        if self.event_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("event canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "ordinal": self.ordinal,
            "operation": self.operation,
            "status": self.status.value,
            "parents": [item.to_dict() for item in self.parents],
            "child": None if self.child is None else self.child.to_dict(),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def create(
        cls,
        generation: int,
        ordinal: int,
        operation: str,
        status: OutcomeStatus,
        parents: Sequence[CandidateArtifact],
        child: CandidateArtifact | None,
        reason_codes: Sequence[str] = (),
    ) -> EvolutionEvent:
        refs = tuple(sorted((item.ref for item in parents), key=lambda item: item.artifact_id))
        child_ref = None if child is None else child.ref
        reasons = tuple(sorted(reason_codes))
        body = {
            "schema_version": "sigma-evolution-event-1.0",
            "generation": generation,
            "ordinal": ordinal,
            "operation": operation,
            "status": status.value,
            "parents": [item.to_dict() for item in refs],
            "child": None if child_ref is None else child_ref.to_dict(),
            "reason_codes": list(reasons),
        }
        return cls(
            generation,
            ordinal,
            operation,
            status,
            refs,
            child_ref,
            reasons,
            canonical_sha256(body),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "event_sha256": self.event_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvolutionEvent:
        _exact_keys(
            value,
            {
                "schema_version",
                "generation",
                "ordinal",
                "operation",
                "status",
                "parents",
                "child",
                "reason_codes",
                "event_sha256",
            },
            "evolution event",
        )
        if not isinstance(value["parents"], list) or not isinstance(value["reason_codes"], list):
            raise SchemaViolation("event collections must be arrays")
        try:
            status = OutcomeStatus(value["status"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("event status is not registered") from error
        return cls(
            value["generation"],
            value["ordinal"],
            str(value["operation"]),
            status,
            tuple(ArtifactRef.from_dict(item) for item in value["parents"]),
            None if value["child"] is None else ArtifactRef.from_dict(value["child"]),
            tuple(value["reason_codes"]),
            str(value["event_sha256"]),
            str(value["schema_version"]),
        )


class MutationOperator(Protocol):
    def __call__(self, parent: CandidateArtifact, stream: SeedStream) -> CandidateArtifact: ...


class CrossoverOperator(Protocol):
    def __call__(
        self, left: CandidateArtifact, right: CandidateArtifact, stream: SeedStream
    ) -> CandidateArtifact: ...


CandidateEvaluator = Callable[[CandidateArtifact], EvaluationOutcome]


@dataclass(frozen=True, slots=True)
class EvolutionRun:
    seed_sha256: str
    budget: EvolutionBudget
    domain_pack: DomainPackRef
    artifacts: tuple[CandidateArtifact, ...]
    lineage: tuple[LineageRecord, ...]
    evaluations: tuple[EvaluationOutcome, ...]
    events: tuple[EvolutionEvent, ...]
    final_population: tuple[ArtifactRef, ...]
    counts: Mapping[str, int]
    content_sha256: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaViolation("evolution run schema_version changed")
        _digest(self.seed_sha256, "seed_sha256")
        clean_counts = _validate_run_collections(
            self.budget,
            self.domain_pack,
            self.artifacts,
            self.lineage,
            self.evaluations,
            self.events,
            self.final_population,
        )
        if dict(self.counts) != clean_counts:
            raise SchemaViolation("evolution run counts changed")
        object.__setattr__(self, "counts", clean_counts)
        _digest(self.content_sha256, "evolution content_sha256")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("evolution run canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed_sha256": self.seed_sha256,
            "budget": self.budget.to_dict(),
            "domain_pack": self.domain_pack.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "lineage": [item.to_dict() for item in self.lineage],
            "evaluations": [item.to_dict() for item in self.evaluations],
            "events": [item.to_dict() for item in self.events],
            "final_population": [item.to_dict() for item in self.final_population],
            "counts": dict(self.counts),
            "selection_scope": "bounded_heuristic_population_search_only",
            "physics_truth_established": False,
            "promotion_allowed": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def create(
        cls,
        seed_sha256: str,
        budget: EvolutionBudget,
        domain_pack: DomainPackRef,
        artifacts: Sequence[CandidateArtifact],
        lineage: Sequence[LineageRecord],
        evaluations: Sequence[EvaluationOutcome],
        events: Sequence[EvolutionEvent],
        final_population: Sequence[ArtifactRef],
    ) -> EvolutionRun:
        artifact_tuple = tuple(artifacts)
        lineage_tuple = tuple(lineage)
        evaluation_tuple = tuple(evaluations)
        event_tuple = tuple(events)
        final_tuple = tuple(final_population)
        counts = _validate_run_collections(
            budget,
            domain_pack,
            artifact_tuple,
            lineage_tuple,
            evaluation_tuple,
            event_tuple,
            final_tuple,
        )
        partial = cls.__new__(cls)
        object.__setattr__(partial, "seed_sha256", seed_sha256)
        object.__setattr__(partial, "budget", budget)
        object.__setattr__(partial, "domain_pack", domain_pack)
        object.__setattr__(partial, "artifacts", artifact_tuple)
        object.__setattr__(partial, "lineage", lineage_tuple)
        object.__setattr__(partial, "evaluations", evaluation_tuple)
        object.__setattr__(partial, "events", event_tuple)
        object.__setattr__(partial, "final_population", final_tuple)
        object.__setattr__(partial, "counts", counts)
        object.__setattr__(partial, "schema_version", SCHEMA_VERSION)
        digest = canonical_sha256(partial._body())
        return cls(
            seed_sha256,
            budget,
            domain_pack,
            artifact_tuple,
            lineage_tuple,
            evaluation_tuple,
            event_tuple,
            final_tuple,
            counts,
            digest,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvolutionRun:
        _exact_keys(
            value,
            {
                "schema_version",
                "seed_sha256",
                "budget",
                "domain_pack",
                "artifacts",
                "lineage",
                "evaluations",
                "events",
                "final_population",
                "counts",
                "selection_scope",
                "physics_truth_established",
                "promotion_allowed",
                "content_sha256",
            },
            "evolution run",
        )
        if (
            value["selection_scope"] != "bounded_heuristic_population_search_only"
            or value["physics_truth_established"] is not False
            or value["promotion_allowed"] is not False
            or not all(
                isinstance(value[key], list)
                for key in ("artifacts", "lineage", "evaluations", "events", "final_population")
            )
        ):
            raise SchemaViolation("evolution run scientific or collection boundary changed")
        return cls(
            str(value["seed_sha256"]),
            EvolutionBudget.from_dict(value["budget"]),
            DomainPackRef.from_dict(value["domain_pack"]),
            tuple(CandidateArtifact.from_dict(item) for item in value["artifacts"]),
            tuple(LineageRecord.from_dict(item) for item in value["lineage"]),
            tuple(EvaluationOutcome.from_dict(item) for item in value["evaluations"]),
            tuple(EvolutionEvent.from_dict(item) for item in value["events"]),
            tuple(ArtifactRef.from_dict(item) for item in value["final_population"]),
            value["counts"],
            str(value["content_sha256"]),
            str(value["schema_version"]),
        )


def _validate_run_collections(
    budget: EvolutionBudget,
    domain_pack: DomainPackRef,
    artifacts: Sequence[CandidateArtifact],
    lineage: Sequence[LineageRecord],
    evaluations: Sequence[EvaluationOutcome],
    events: Sequence[EvolutionEvent],
    final_population: Sequence[ArtifactRef],
) -> dict[str, int]:
    artifact_map: dict[str, CandidateArtifact] = {}
    for artifact in artifacts:
        artifact.validate()
        if artifact.provenance.domain_pack != domain_pack:
            raise SchemaViolation("evolution artifact domain pack changed")
        if artifact.content_sha256 in artifact_map:
            raise SchemaViolation("evolution artifacts are not content-hash deduplicated")
        artifact_map[artifact.content_sha256] = artifact
    if tuple(sorted(artifact_map)) != tuple(item.content_sha256 for item in artifacts):
        raise SchemaViolation("evolution artifacts must be content-hash sorted")
    if len(lineage) != len(artifacts):
        raise SchemaViolation("every evolution artifact requires exactly one lineage record")
    lineage_map = {item.child.content_sha256: item for item in lineage}
    if len(lineage_map) != len(lineage) or set(lineage_map) != set(artifact_map):
        raise SchemaViolation("lineage does not cover the canonical artifact set")
    for digest, record in lineage_map.items():
        artifact = artifact_map[digest]
        if record.child != artifact.ref:
            raise SchemaViolation("lineage child reference changed")
        if record.generation > budget.generations:
            raise SchemaViolation("lineage generation budget exceeded")
        ordinal_bound = (
            budget.population_size
            if record.operation == "seed"
            else budget.offspring_per_generation
        )
        if record.ordinal >= ordinal_bound:
            raise SchemaViolation("lineage ordinal budget exceeded")
        expected_inputs = () if record.operation == "seed" else record.parents
        if artifact.provenance.inputs != expected_inputs:
            raise SchemaViolation("candidate provenance inputs do not match lineage parents")
        for parent in record.parents:
            parent_artifact = artifact_map.get(parent.content_sha256)
            if parent_artifact is None or parent_artifact.ref != parent:
                raise SchemaViolation("lineage parent is outside the canonical artifact set")
            if lineage_map[parent.content_sha256].generation >= record.generation:
                raise SchemaViolation("lineage parent is not from an earlier generation")
    evaluation_map = {item.artifact.content_sha256: item for item in evaluations}
    if len(evaluation_map) != len(evaluations) or set(evaluation_map) != set(artifact_map):
        raise SchemaViolation("evaluations do not cover the canonical artifact set")
    if len(evaluations) > budget.max_evaluations:
        raise SchemaViolation("evaluation budget exceeded")
    for digest, evaluation in evaluation_map.items():
        if evaluation.artifact != artifact_map[digest].ref:
            raise SchemaViolation("evaluation artifact reference changed")
        matching_events = [
            event
            for event in events
            if event.child == artifact_map[digest].ref
            and event.status is evaluation.status
            and event.reason_codes == evaluation.reason_codes
        ]
        if len(matching_events) != 1:
            raise SchemaViolation("evaluation does not have one exact acceptance event")
        accepted_event = matching_events[0]
        record = lineage_map[digest]
        if (
            accepted_event.operation != record.operation
            or accepted_event.parents != record.parents
            or accepted_event.generation != record.generation
            or accepted_event.ordinal != record.ordinal
        ):
            raise SchemaViolation("acceptance event does not match lineage")
    final_hashes = tuple(item.content_sha256 for item in final_population)
    if len(set(final_hashes)) != len(final_hashes) or len(final_hashes) > budget.population_size:
        raise SchemaViolation("final population boundary changed")
    if any(
        digest not in artifact_map
        or final_population[index] != artifact_map[digest].ref
        or evaluation_map[digest].status is not OutcomeStatus.PASS
        for index, digest in enumerate(final_hashes)
    ):
        raise SchemaViolation("final population contains an ineligible artifact")
    known_refs = {artifact.ref for artifact in artifacts}
    for event in events:
        if event.generation > budget.generations:
            raise SchemaViolation("event generation budget exceeded")
        ordinal_bound = (
            budget.population_size if event.operation == "seed" else budget.offspring_per_generation
        )
        if event.ordinal >= ordinal_bound or any(
            parent not in known_refs for parent in event.parents
        ):
            raise SchemaViolation("event ordinal or parent boundary changed")
    if len(events) > budget.population_size + budget.max_operator_attempts:
        raise SchemaViolation("operator attempt budget exceeded")
    return {
        "artifacts": len(artifacts),
        "lineage_records": len(lineage),
        "evaluations": len(evaluations),
        "operator_attempts": sum(item.operation != "seed" for item in events),
        "deduplicated": sum("duplicate_content_hash" in item.reason_codes for item in events),
        "pass": sum(item.status is OutcomeStatus.PASS for item in evaluations),
        "block": sum(item.status is OutcomeStatus.BLOCK for item in evaluations),
        "reject": sum(item.status is OutcomeStatus.REJECT for item in evaluations),
        "error": sum(item.status is OutcomeStatus.ERROR for item in evaluations),
        "event_block": sum(item.status is OutcomeStatus.BLOCK for item in events),
        "event_reject": sum(item.status is OutcomeStatus.REJECT for item in events),
        "event_error": sum(item.status is OutcomeStatus.ERROR for item in events),
        "final_population": len(final_population),
    }


def _safe_evaluate(artifact: CandidateArtifact, evaluator: CandidateEvaluator) -> EvaluationOutcome:
    try:
        outcome = evaluator(artifact)
    except Exception:  # noqa: BLE001 - untrusted bounded callback is an error outcome
        return EvaluationOutcome.create(
            artifact, OutcomeStatus.ERROR, reason_codes=("evaluator_exception",)
        )
    if not isinstance(outcome, EvaluationOutcome) or outcome.artifact != artifact.ref:
        return EvaluationOutcome.create(
            artifact, OutcomeStatus.ERROR, reason_codes=("invalid_evaluator_outcome",)
        )
    return outcome


def _child_error(
    candidate: Any,
    parents: Sequence[CandidateArtifact],
    domain_pack: DomainPackRef,
) -> tuple[CandidateArtifact | None, OutcomeStatus | None, tuple[str, ...]]:
    if not isinstance(candidate, CandidateArtifact):
        return None, OutcomeStatus.REJECT, ("operator_return_not_candidate",)
    try:
        candidate.validate()
    except SchemaViolation:
        return None, OutcomeStatus.REJECT, ("candidate_identity_invalid",)
    if candidate.provenance.domain_pack != domain_pack:
        return candidate, OutcomeStatus.REJECT, ("domain_pack_mismatch",)
    if len({parent.kind for parent in parents}) != 1 or candidate.kind != parents[0].kind:
        return candidate, OutcomeStatus.REJECT, ("artifact_kind_mismatch",)
    expected = tuple(sorted((parent.ref for parent in parents), key=lambda item: item.artifact_id))
    if candidate.provenance.inputs != expected:
        return candidate, OutcomeStatus.REJECT, ("parent_lineage_mismatch",)
    return candidate, None, ()


def evolve_candidates(
    initial_population: Sequence[CandidateArtifact],
    *,
    seed: str,
    budget: EvolutionBudget,
    mutate: MutationOperator,
    crossover: CrossoverOperator,
    evaluate: CandidateEvaluator,
) -> EvolutionRun:
    """Run deterministic bounded selection without granting scientific promotion."""

    if not initial_population:
        raise SchemaViolation("initial population must be nonempty")
    if len(initial_population) > budget.population_size:
        raise SchemaViolation("initial population exceeds population_size")
    if len(initial_population) > budget.max_evaluations:
        raise SchemaViolation("initial population exceeds evaluation budget")
    for artifact in initial_population:
        if not isinstance(artifact, CandidateArtifact):
            raise SchemaViolation("initial population contains a non-candidate")
        artifact.validate()
    domain_pack = initial_population[0].provenance.domain_pack
    if any(item.provenance.domain_pack != domain_pack for item in initial_population):
        raise SchemaViolation("initial population spans multiple domain packs")

    stream = SeedStream(seed)
    artifacts: dict[str, CandidateArtifact] = {}
    lineage: list[LineageRecord] = []
    evaluations: dict[str, EvaluationOutcome] = {}
    events: list[EvolutionEvent] = []
    for ordinal, artifact in enumerate(initial_population):
        digest = artifact.content_sha256
        if digest in artifacts:
            events.append(
                EvolutionEvent.create(
                    0,
                    ordinal,
                    "seed",
                    OutcomeStatus.BLOCK,
                    (),
                    artifact,
                    ("duplicate_content_hash",),
                )
            )
            continue
        artifacts[digest] = artifact
        lineage.append(LineageRecord.create(0, ordinal, "seed", artifact, ()))
        outcome = _safe_evaluate(artifact, evaluate)
        evaluations[digest] = outcome
        events.append(
            EvolutionEvent.create(
                0,
                ordinal,
                "seed",
                outcome.status,
                (),
                artifact,
                outcome.reason_codes,
            )
        )

    def eligible() -> list[CandidateArtifact]:
        return sorted(
            (
                artifact
                for digest, artifact in artifacts.items()
                if evaluations[digest].status is OutcomeStatus.PASS
            ),
            key=lambda artifact: (
                -int(evaluations[artifact.content_sha256].score or 0),
                artifact.content_sha256,
            ),
        )[: budget.population_size]

    population = eligible()
    exhausted = False
    for generation in range(1, budget.generations + 1):
        if not population or exhausted:
            break
        for ordinal in range(budget.offspring_per_generation):
            if len(evaluations) >= budget.max_evaluations:
                exhausted = True
                break
            use_crossover = ordinal % 2 == 1 and len(population) >= 2
            operation = "crossover" if use_crossover else "mutation"
            if use_crossover:
                left = stream.choose(population)
                remaining = [item for item in population if item.ref != left.ref]
                right = stream.choose(remaining)
                parents = (left, right)
                try:
                    raw_child: Any = crossover(left, right, stream)
                except Exception:  # noqa: BLE001 - untrusted bounded callback
                    events.append(
                        EvolutionEvent.create(
                            generation,
                            ordinal,
                            operation,
                            OutcomeStatus.ERROR,
                            parents,
                            None,
                            ("operator_exception",),
                        )
                    )
                    continue
            else:
                parent = stream.choose(population)
                parents = (parent,)
                try:
                    raw_child = mutate(parent, stream)
                except Exception:  # noqa: BLE001 - untrusted bounded callback
                    events.append(
                        EvolutionEvent.create(
                            generation,
                            ordinal,
                            operation,
                            OutcomeStatus.ERROR,
                            parents,
                            None,
                            ("operator_exception",),
                        )
                    )
                    continue
            child, failure_status, reasons = _child_error(raw_child, parents, domain_pack)
            if failure_status is not None:
                events.append(
                    EvolutionEvent.create(
                        generation,
                        ordinal,
                        operation,
                        failure_status,
                        parents,
                        child,
                        reasons,
                    )
                )
                continue
            assert child is not None
            if child.content_sha256 in artifacts:
                events.append(
                    EvolutionEvent.create(
                        generation,
                        ordinal,
                        operation,
                        OutcomeStatus.BLOCK,
                        parents,
                        child,
                        ("duplicate_content_hash",),
                    )
                )
                continue
            artifacts[child.content_sha256] = child
            lineage.append(LineageRecord.create(generation, ordinal, operation, child, parents))
            outcome = _safe_evaluate(child, evaluate)
            evaluations[child.content_sha256] = outcome
            events.append(
                EvolutionEvent.create(
                    generation,
                    ordinal,
                    operation,
                    outcome.status,
                    parents,
                    child,
                    outcome.reason_codes,
                )
            )
        population = eligible()

    ordered_artifacts = tuple(artifacts[key] for key in sorted(artifacts))
    lineage_by_child = {item.child.content_sha256: item for item in lineage}
    ordered_lineage = tuple(lineage_by_child[item.content_sha256] for item in ordered_artifacts)
    ordered_evaluations = tuple(evaluations[item.content_sha256] for item in ordered_artifacts)
    final = tuple(item.ref for item in eligible())
    return EvolutionRun.create(
        hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        budget,
        domain_pack,
        ordered_artifacts,
        ordered_lineage,
        ordered_evaluations,
        events,
        final,
    )
