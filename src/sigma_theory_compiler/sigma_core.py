"""Domain-independent, fail-closed foundations for Sigma candidate evaluation.

The module deliberately contains no subject-specific vocabulary or imports. Domain packs describe
their own stages and gates, while Sigma Core owns canonical serialization, candidate/provenance
identity, typed outcomes, and the promotion hash chain.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = "sigma-core-1.0"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SigmaCoreError(ValueError):
    """Base class for a closed-world Sigma Core validation failure."""


class SchemaViolation(SigmaCoreError):
    """A value is outside an exact schema or canonical-value boundary."""


class DomainPackViolation(SigmaCoreError):
    """A domain pack or one of its returned outcomes violates its descriptor."""


class PromotionDenied(SigmaCoreError):
    """A fail-closed promotion precondition was not exactly established."""


class ArtifactKind(str, Enum):
    FORMULA = "formula"
    IDENTITY = "identity"
    CONJECTURE = "conjecture"
    THEOREM = "theorem"
    PROOF = "proof"
    ALGORITHM = "algorithm"
    CONSTRUCTION = "construction"
    PHYSICAL_ACTION = "physical_action"


class OutcomeStatus(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    REJECT = "reject"
    ERROR = "error"


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} must match {_IDENTIFIER.pattern}")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaViolation(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SchemaViolation(f"{label} must be nonempty and stripped")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SchemaViolation(f"{label} keys changed")


def _unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(set(result)) != len(result):
        raise SchemaViolation(f"{label} contains duplicates")
    return result


def _json_value(value: Any, path: str = "$", *, require_object: bool = False) -> Any:
    """Return a detached canonical-JSON value, rejecting ambiguous numeric values.

    Floats are intentionally excluded.  Integers, decimal strings, or a domain pack's own
    exact representation avoid NaN/infinity and cross-runtime floating serialization drift.
    """

    if value is None or isinstance(value, (str, bool, int)):
        result = value
    elif isinstance(value, float):
        raise SchemaViolation(f"floating value forbidden at {path}")
    elif isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise SchemaViolation(f"non-string object key at {path}")
            result[key] = _json_value(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = [_json_value(child, f"{path}[{index}]") for index, child in enumerate(value)]
    else:
        raise SchemaViolation(f"non-JSON value at {path}: {type(value).__name__}")
    if require_object and not isinstance(result, dict):
        raise SchemaViolation(f"{path} must be a JSON object")
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value using Sigma Core's deterministic canonical JSON subset."""

    clean = _json_value(value)
    return json.dumps(
        clean, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    content_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.artifact_id, "artifact_id")
        _hash(self.content_sha256, "artifact content_sha256")

    def to_dict(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        _exact_keys(value, {"artifact_id", "content_sha256"}, "artifact reference")
        return cls(str(value["artifact_id"]), str(value["content_sha256"]))


@dataclass(frozen=True, slots=True)
class SourceBinding:
    role: str
    path: str
    file_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.role, "source role")
        _hash(self.file_sha256, "source file_sha256")
        if not isinstance(self.path, str) or not self.path or "\\" in self.path:
            raise SchemaViolation("source path must be a nonempty POSIX relative path")
        parsed = PurePosixPath(self.path)
        if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
            raise SchemaViolation("source path must stay within its project root")
        if parsed.as_posix() != self.path:
            raise SchemaViolation("source path is not canonical POSIX form")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "path": self.path, "file_sha256": self.file_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceBinding:
        _exact_keys(value, {"role", "path", "file_sha256"}, "source binding")
        return cls(str(value["role"]), str(value["path"]), str(value["file_sha256"]))


@dataclass(frozen=True, slots=True)
class DomainPackRef:
    pack_id: str
    pack_version: str
    descriptor_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.pack_id, "pack_id")
        _nonempty(self.pack_version, "pack_version")
        _hash(self.descriptor_sha256, "descriptor_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "descriptor_sha256": self.descriptor_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DomainPackRef:
        _exact_keys(
            value, {"pack_id", "pack_version", "descriptor_sha256"}, "domain pack reference"
        )
        return cls(
            str(value["pack_id"]),
            str(value["pack_version"]),
            str(value["descriptor_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    domain_pack: DomainPackRef
    parameters_sha256: str
    inputs: tuple[ArtifactRef, ...] = ()
    sources: tuple[SourceBinding, ...] = ()
    schema_version: str = "sigma-core-provenance-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-core-provenance-1.0":
            raise SchemaViolation("provenance schema_version changed")
        _hash(self.parameters_sha256, "parameters_sha256")
        if len({item.artifact_id for item in self.inputs}) != len(self.inputs):
            raise SchemaViolation("provenance input artifact IDs contain duplicates")
        if len({item.role for item in self.sources}) != len(self.sources):
            raise SchemaViolation("provenance source roles contain duplicates")
        if tuple(sorted(self.inputs, key=lambda item: item.artifact_id)) != self.inputs:
            raise SchemaViolation("provenance inputs must be sorted by artifact_id")
        if tuple(sorted(self.sources, key=lambda item: item.role)) != self.sources:
            raise SchemaViolation("provenance sources must be sorted by role")

    @classmethod
    def create(
        cls,
        domain_pack: DomainPackRef,
        parameters: Mapping[str, Any],
        *,
        inputs: Sequence[ArtifactRef] = (),
        sources: Sequence[SourceBinding] = (),
    ) -> ProvenanceRecord:
        return cls(
            domain_pack=domain_pack,
            parameters_sha256=canonical_sha256(_json_value(parameters, require_object=True)),
            inputs=tuple(sorted(inputs, key=lambda item: item.artifact_id)),
            sources=tuple(sorted(sources, key=lambda item: item.role)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain_pack": self.domain_pack.to_dict(),
            "parameters_sha256": self.parameters_sha256,
            "inputs": [item.to_dict() for item in self.inputs],
            "sources": [item.to_dict() for item in self.sources],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProvenanceRecord:
        _exact_keys(
            value,
            {"schema_version", "domain_pack", "parameters_sha256", "inputs", "sources"},
            "provenance",
        )
        if not isinstance(value["inputs"], list) or not isinstance(value["sources"], list):
            raise SchemaViolation("provenance inputs and sources must be arrays")
        return cls(
            domain_pack=DomainPackRef.from_dict(value["domain_pack"]),
            parameters_sha256=str(value["parameters_sha256"]),
            inputs=tuple(ArtifactRef.from_dict(item) for item in value["inputs"]),
            sources=tuple(SourceBinding.from_dict(item) for item in value["sources"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    artifact_id: str
    kind: ArtifactKind
    statement: str
    representation: Mapping[str, Any]
    assumptions: tuple[str, ...]
    claims: tuple[str, ...]
    provenance: ProvenanceRecord
    content_sha256: str
    schema_version: str = "sigma-core-candidate-artifact-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-core-candidate-artifact-1.0":
            raise SchemaViolation("candidate schema_version changed")
        _identifier(self.artifact_id, "artifact_id")
        _nonempty(self.statement, "statement")
        clean = _json_value(self.representation, "$.representation", require_object=True)
        object.__setattr__(self, "representation", clean)
        assumptions = tuple(_nonempty(item, "assumption") for item in self.assumptions)
        claims = tuple(_identifier(item, "claim") for item in self.claims)
        if assumptions != tuple(sorted(set(assumptions))):
            raise SchemaViolation("assumptions must be unique and sorted")
        if claims != tuple(sorted(set(claims))):
            raise SchemaViolation("claims must be unique and sorted")
        _hash(self.content_sha256, "candidate content_sha256")
        expected = canonical_sha256(self._body())
        if self.content_sha256 != expected or self.artifact_id != f"sig-{expected[:24]}":
            raise SchemaViolation("candidate canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "statement": self.statement,
            "representation": _json_value(
                self.representation, "$.representation", require_object=True
            ),
            "assumptions": list(self.assumptions),
            "claims": list(self.claims),
            "provenance": self.provenance.to_dict(),
        }

    def validate(self) -> None:
        """Recheck canonical identity after crossing a potentially mutable caller boundary."""

        expected = canonical_sha256(self._body())
        if self.content_sha256 != expected or self.artifact_id != f"sig-{expected[:24]}":
            raise SchemaViolation("candidate canonical identity changed")

    @classmethod
    def create(
        cls,
        kind: ArtifactKind,
        statement: str,
        representation: Mapping[str, Any],
        provenance: ProvenanceRecord,
        *,
        assumptions: Sequence[str] = (),
        claims: Sequence[str] = (),
    ) -> CandidateArtifact:
        body = {
            "schema_version": "sigma-core-candidate-artifact-1.0",
            "kind": kind.value,
            "statement": _nonempty(statement, "statement"),
            "representation": _json_value(representation, "$.representation", require_object=True),
            "assumptions": sorted(_unique(tuple(assumptions), "assumptions")),
            "claims": sorted(_unique(tuple(claims), "claims")),
            "provenance": provenance.to_dict(),
        }
        digest = canonical_sha256(body)
        return cls(
            artifact_id=f"sig-{digest[:24]}",
            kind=kind,
            statement=body["statement"],
            representation=body["representation"],
            assumptions=tuple(body["assumptions"]),
            claims=tuple(body["claims"]),
            provenance=provenance,
            content_sha256=digest,
        )

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(self.artifact_id, self.content_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            **self._body(),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateArtifact:
        _exact_keys(
            value,
            {
                "artifact_id",
                "schema_version",
                "kind",
                "statement",
                "representation",
                "assumptions",
                "claims",
                "provenance",
                "content_sha256",
            },
            "candidate artifact",
        )
        if not isinstance(value["assumptions"], list) or not isinstance(value["claims"], list):
            raise SchemaViolation("candidate assumptions and claims must be arrays")
        try:
            kind = ArtifactKind(value["kind"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("candidate artifact kind is not registered") from error
        return cls(
            artifact_id=str(value["artifact_id"]),
            kind=kind,
            statement=str(value["statement"]),
            representation=value["representation"],
            assumptions=tuple(value["assumptions"]),
            claims=tuple(value["claims"]),
            provenance=ProvenanceRecord.from_dict(value["provenance"]),
            content_sha256=str(value["content_sha256"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class StageDefinition:
    stage_id: str
    ordinal: int
    allowed_kinds: tuple[ArtifactKind, ...]
    prerequisites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.stage_id, "stage_id")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise SchemaViolation("stage ordinal must be a nonnegative integer")
        if not self.allowed_kinds or len(set(self.allowed_kinds)) != len(self.allowed_kinds):
            raise SchemaViolation("stage allowed_kinds must be nonempty and unique")
        if tuple(sorted(self.allowed_kinds, key=lambda item: item.value)) != self.allowed_kinds:
            raise SchemaViolation("stage allowed_kinds must be sorted")
        prerequisites = tuple(
            _identifier(item, "stage prerequisite") for item in self.prerequisites
        )
        if prerequisites != tuple(sorted(set(prerequisites))):
            raise SchemaViolation("stage prerequisites must be unique and sorted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "ordinal": self.ordinal,
            "allowed_kinds": [item.value for item in self.allowed_kinds],
            "prerequisites": list(self.prerequisites),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StageDefinition:
        _exact_keys(
            value,
            {"stage_id", "ordinal", "allowed_kinds", "prerequisites"},
            "stage definition",
        )
        if not isinstance(value["allowed_kinds"], list) or not isinstance(
            value["prerequisites"], list
        ):
            raise SchemaViolation("stage definition collection fields must be arrays")
        try:
            kinds = tuple(ArtifactKind(item) for item in value["allowed_kinds"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("stage contains an unregistered artifact kind") from error
        return cls(str(value["stage_id"]), value["ordinal"], kinds, tuple(value["prerequisites"]))


@dataclass(frozen=True, slots=True)
class GateDefinition:
    gate_id: str
    from_stage: str | None
    to_stage: str
    allowed_kinds: tuple[ArtifactKind, ...]
    required_stages: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.gate_id, "gate_id")
        if self.from_stage is not None:
            _identifier(self.from_stage, "gate from_stage")
        _identifier(self.to_stage, "gate to_stage")
        if not self.allowed_kinds or len(set(self.allowed_kinds)) != len(self.allowed_kinds):
            raise SchemaViolation("gate allowed_kinds must be nonempty and unique")
        if tuple(sorted(self.allowed_kinds, key=lambda item: item.value)) != self.allowed_kinds:
            raise SchemaViolation("gate allowed_kinds must be sorted")
        required = tuple(_identifier(item, "required stage") for item in self.required_stages)
        if required != tuple(sorted(set(required))) or self.to_stage not in required:
            raise SchemaViolation("required_stages must be unique, sorted, and include to_stage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "allowed_kinds": [item.value for item in self.allowed_kinds],
            "required_stages": list(self.required_stages),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GateDefinition:
        _exact_keys(
            value,
            {"gate_id", "from_stage", "to_stage", "allowed_kinds", "required_stages"},
            "gate definition",
        )
        if not isinstance(value["allowed_kinds"], list) or not isinstance(
            value["required_stages"], list
        ):
            raise SchemaViolation("gate definition collection fields must be arrays")
        if value["from_stage"] is not None and not isinstance(value["from_stage"], str):
            raise SchemaViolation("gate from_stage must be null or a stage ID")
        try:
            kinds = tuple(ArtifactKind(item) for item in value["allowed_kinds"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("gate contains an unregistered artifact kind") from error
        return cls(
            str(value["gate_id"]),
            value["from_stage"],
            str(value["to_stage"]),
            kinds,
            tuple(value["required_stages"]),
        )


@dataclass(frozen=True, slots=True)
class DomainPackDescriptor:
    pack_id: str
    pack_version: str
    supported_kinds: tuple[ArtifactKind, ...]
    stages: tuple[StageDefinition, ...]
    gates: tuple[GateDefinition, ...]
    schema_version: str = "sigma-core-domain-pack-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-core-domain-pack-1.0":
            raise SchemaViolation("domain pack schema_version changed")
        _identifier(self.pack_id, "pack_id")
        _nonempty(self.pack_version, "pack_version")
        if (
            not self.supported_kinds
            or tuple(sorted(set(self.supported_kinds), key=lambda item: item.value))
            != self.supported_kinds
        ):
            raise SchemaViolation("supported_kinds must be nonempty, unique, and sorted")
        if not self.stages or tuple(stage.ordinal for stage in self.stages) != tuple(
            range(len(self.stages))
        ):
            raise SchemaViolation("stages must have consecutive tuple-order ordinals")
        stage_map = {stage.stage_id: stage for stage in self.stages}
        if len(stage_map) != len(self.stages):
            raise SchemaViolation("stage IDs contain duplicates")
        for stage in self.stages:
            if not set(stage.allowed_kinds) <= set(self.supported_kinds):
                raise SchemaViolation("stage permits unsupported artifact kind")
            for prerequisite in stage.prerequisites:
                if (
                    prerequisite not in stage_map
                    or stage_map[prerequisite].ordinal >= stage.ordinal
                ):
                    raise SchemaViolation("stage prerequisite must be an earlier registered stage")
        gate_map = {gate.gate_id: gate for gate in self.gates}
        if len(gate_map) != len(self.gates) or tuple(sorted(gate_map)) != tuple(
            gate.gate_id for gate in self.gates
        ):
            raise SchemaViolation("gates must have unique IDs and sorted tuple order")
        for gate in self.gates:
            if gate.to_stage not in stage_map or (
                gate.from_stage is not None and gate.from_stage not in stage_map
            ):
                raise SchemaViolation("gate references an unregistered stage")
            if gate.from_stage is not None and (
                stage_map[gate.from_stage].ordinal >= stage_map[gate.to_stage].ordinal
            ):
                raise SchemaViolation("gate must move strictly forward")
            required_closure = {gate.to_stage, *stage_map[gate.to_stage].prerequisites}
            if not required_closure <= set(gate.required_stages):
                raise SchemaViolation("gate omits a target-stage prerequisite")
            if any(
                stage_id not in stage_map
                or stage_map[stage_id].ordinal > stage_map[gate.to_stage].ordinal
                for stage_id in gate.required_stages
            ):
                raise SchemaViolation("gate required_stages boundary changed")
            if not set(gate.allowed_kinds) <= set(stage_map[gate.to_stage].allowed_kinds):
                raise SchemaViolation("gate permits kind forbidden by its target stage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "supported_kinds": [item.value for item in self.supported_kinds],
            "stages": [item.to_dict() for item in self.stages],
            "gates": [item.to_dict() for item in self.gates],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DomainPackDescriptor:
        _exact_keys(
            value,
            {"schema_version", "pack_id", "pack_version", "supported_kinds", "stages", "gates"},
            "domain pack descriptor",
        )
        if not all(isinstance(value[key], list) for key in ("supported_kinds", "stages", "gates")):
            raise SchemaViolation("domain pack descriptor collection fields must be arrays")
        try:
            kinds = tuple(ArtifactKind(item) for item in value["supported_kinds"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("domain pack contains an unregistered artifact kind") from error
        return cls(
            str(value["pack_id"]),
            str(value["pack_version"]),
            kinds,
            tuple(StageDefinition.from_dict(item) for item in value["stages"]),
            tuple(GateDefinition.from_dict(item) for item in value["gates"]),
            str(value["schema_version"]),
        )

    @property
    def ref(self) -> DomainPackRef:
        return DomainPackRef(self.pack_id, self.pack_version, canonical_sha256(self.to_dict()))

    def stage(self, stage_id: str) -> StageDefinition:
        try:
            return next(item for item in self.stages if item.stage_id == stage_id)
        except StopIteration as error:
            raise DomainPackViolation(f"unknown stage: {stage_id}") from error

    def gate(self, gate_id: str) -> GateDefinition:
        try:
            return next(item for item in self.gates if item.gate_id == gate_id)
        except StopIteration as error:
            raise DomainPackViolation(f"unknown gate: {gate_id}") from error


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    passed: bool
    details_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.check_id, "check_id")
        if type(self.passed) is not bool:
            raise SchemaViolation("check passed must be a boolean")
        _hash(self.details_sha256, "check details_sha256")

    @classmethod
    def create(cls, check_id: str, passed: bool, details: Mapping[str, Any]) -> CheckResult:
        return cls(check_id, passed, canonical_sha256(_json_value(details, require_object=True)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "details_sha256": self.details_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckResult:
        _exact_keys(value, {"check_id", "passed", "details_sha256"}, "check result")
        return cls(str(value["check_id"]), value["passed"], str(value["details_sha256"]))


@dataclass(frozen=True, slots=True)
class OutcomeRef:
    outcome_id: str
    outcome_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.outcome_id, "outcome_id")
        _hash(self.outcome_sha256, "outcome_sha256")

    def to_dict(self) -> dict[str, str]:
        return {"outcome_id": self.outcome_id, "outcome_sha256": self.outcome_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OutcomeRef:
        _exact_keys(value, {"outcome_id", "outcome_sha256"}, "outcome reference")
        return cls(str(value["outcome_id"]), str(value["outcome_sha256"]))


def _validate_outcome_parts(
    status: OutcomeStatus,
    checks: tuple[CheckResult, ...],
    reason_codes: tuple[str, ...],
) -> None:
    if not checks or tuple(sorted(checks, key=lambda item: item.check_id)) != checks:
        raise SchemaViolation("outcome checks must be nonempty, unique, and sorted")
    if len({item.check_id for item in checks}) != len(checks):
        raise SchemaViolation("outcome check IDs contain duplicates")
    reasons = tuple(_identifier(item, "reason code") for item in reason_codes)
    if reasons != tuple(sorted(set(reasons))):
        raise SchemaViolation("reason_codes must be unique and sorted")
    if status is OutcomeStatus.PASS:
        if reasons or not all(item.passed for item in checks):
            raise SchemaViolation("pass outcome requires all checks passing and no reason codes")
    elif not reasons or all(item.passed for item in checks):
        raise SchemaViolation("non-pass outcome requires reasons and a failed check")


@dataclass(frozen=True, slots=True)
class StageOutcome:
    stage_id: str
    artifact: ArtifactRef
    status: OutcomeStatus
    checks: tuple[CheckResult, ...]
    evidence: tuple[SourceBinding, ...]
    reason_codes: tuple[str, ...]
    outcome_sha256: str
    schema_version: str = "sigma-core-stage-outcome-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-core-stage-outcome-1.0":
            raise SchemaViolation("stage outcome schema_version changed")
        _identifier(self.stage_id, "stage_id")
        _validate_outcome_parts(self.status, self.checks, self.reason_codes)
        if tuple(sorted(self.evidence, key=lambda item: item.role)) != self.evidence or len(
            {item.role for item in self.evidence}
        ) != len(self.evidence):
            raise SchemaViolation("stage evidence must have unique sorted roles")
        _hash(self.outcome_sha256, "stage outcome_sha256")
        if self.outcome_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("stage outcome canonical hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "artifact": self.artifact.to_dict(),
            "status": self.status.value,
            "checks": [item.to_dict() for item in self.checks],
            "evidence": [item.to_dict() for item in self.evidence],
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def create(
        cls,
        stage_id: str,
        artifact: ArtifactRef,
        status: OutcomeStatus,
        checks: Sequence[CheckResult],
        *,
        evidence: Sequence[SourceBinding] = (),
        reason_codes: Sequence[str] = (),
    ) -> StageOutcome:
        body = {
            "schema_version": "sigma-core-stage-outcome-1.0",
            "stage_id": stage_id,
            "artifact": artifact.to_dict(),
            "status": status.value,
            "checks": [item.to_dict() for item in sorted(checks, key=lambda item: item.check_id)],
            "evidence": [item.to_dict() for item in sorted(evidence, key=lambda item: item.role)],
            "reason_codes": sorted(reason_codes),
        }
        return cls(
            stage_id,
            artifact,
            status,
            tuple(CheckResult.from_dict(item) for item in body["checks"]),
            tuple(SourceBinding.from_dict(item) for item in body["evidence"]),
            tuple(body["reason_codes"]),
            canonical_sha256(body),
        )

    @property
    def ref(self) -> OutcomeRef:
        return OutcomeRef(self.stage_id, self.outcome_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "outcome_sha256": self.outcome_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StageOutcome:
        _exact_keys(
            value,
            {
                "schema_version",
                "stage_id",
                "artifact",
                "status",
                "checks",
                "evidence",
                "reason_codes",
                "outcome_sha256",
            },
            "stage outcome",
        )
        if not all(isinstance(value[key], list) for key in ("checks", "evidence", "reason_codes")):
            raise SchemaViolation("stage outcome collection fields must be arrays")
        try:
            status = OutcomeStatus(value["status"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("unregistered stage outcome status") from error
        return cls(
            str(value["stage_id"]),
            ArtifactRef.from_dict(value["artifact"]),
            status,
            tuple(CheckResult.from_dict(item) for item in value["checks"]),
            tuple(SourceBinding.from_dict(item) for item in value["evidence"]),
            tuple(value["reason_codes"]),
            str(value["outcome_sha256"]),
            str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class GateOutcome:
    gate_id: str
    artifact: ArtifactRef
    status: OutcomeStatus
    stage_outcomes: tuple[OutcomeRef, ...]
    checks: tuple[CheckResult, ...]
    evidence: tuple[SourceBinding, ...]
    reason_codes: tuple[str, ...]
    outcome_sha256: str
    schema_version: str = "sigma-core-gate-outcome-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-core-gate-outcome-1.0":
            raise SchemaViolation("gate outcome schema_version changed")
        _identifier(self.gate_id, "gate_id")
        _validate_outcome_parts(self.status, self.checks, self.reason_codes)
        if (
            tuple(sorted(self.stage_outcomes, key=lambda item: item.outcome_id))
            != self.stage_outcomes
        ):
            raise SchemaViolation("gate stage outcomes must be sorted")
        if len({item.outcome_id for item in self.stage_outcomes}) != len(self.stage_outcomes):
            raise SchemaViolation("gate stage outcomes contain duplicate IDs")
        if tuple(sorted(self.evidence, key=lambda item: item.role)) != self.evidence or len(
            {item.role for item in self.evidence}
        ) != len(self.evidence):
            raise SchemaViolation("gate evidence must have unique sorted roles")
        _hash(self.outcome_sha256, "gate outcome_sha256")
        if self.outcome_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("gate outcome canonical hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate_id": self.gate_id,
            "artifact": self.artifact.to_dict(),
            "status": self.status.value,
            "stage_outcomes": [item.to_dict() for item in self.stage_outcomes],
            "checks": [item.to_dict() for item in self.checks],
            "evidence": [item.to_dict() for item in self.evidence],
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def create(
        cls,
        gate_id: str,
        artifact: ArtifactRef,
        status: OutcomeStatus,
        stage_outcomes: Sequence[OutcomeRef],
        checks: Sequence[CheckResult],
        *,
        evidence: Sequence[SourceBinding] = (),
        reason_codes: Sequence[str] = (),
    ) -> GateOutcome:
        body = {
            "schema_version": "sigma-core-gate-outcome-1.0",
            "gate_id": gate_id,
            "artifact": artifact.to_dict(),
            "status": status.value,
            "stage_outcomes": [
                item.to_dict() for item in sorted(stage_outcomes, key=lambda item: item.outcome_id)
            ],
            "checks": [item.to_dict() for item in sorted(checks, key=lambda item: item.check_id)],
            "evidence": [item.to_dict() for item in sorted(evidence, key=lambda item: item.role)],
            "reason_codes": sorted(reason_codes),
        }
        return cls(
            gate_id,
            artifact,
            status,
            tuple(OutcomeRef.from_dict(item) for item in body["stage_outcomes"]),
            tuple(CheckResult.from_dict(item) for item in body["checks"]),
            tuple(SourceBinding.from_dict(item) for item in body["evidence"]),
            tuple(body["reason_codes"]),
            canonical_sha256(body),
        )

    @property
    def ref(self) -> OutcomeRef:
        return OutcomeRef(self.gate_id, self.outcome_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "outcome_sha256": self.outcome_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GateOutcome:
        _exact_keys(
            value,
            {
                "schema_version",
                "gate_id",
                "artifact",
                "status",
                "stage_outcomes",
                "checks",
                "evidence",
                "reason_codes",
                "outcome_sha256",
            },
            "gate outcome",
        )
        if not all(
            isinstance(value[key], list)
            for key in ("stage_outcomes", "checks", "evidence", "reason_codes")
        ):
            raise SchemaViolation("gate outcome collection fields must be arrays")
        try:
            status = OutcomeStatus(value["status"])
        except (TypeError, ValueError) as error:
            raise SchemaViolation("unregistered gate outcome status") from error
        return cls(
            str(value["gate_id"]),
            ArtifactRef.from_dict(value["artifact"]),
            status,
            tuple(OutcomeRef.from_dict(item) for item in value["stage_outcomes"]),
            tuple(CheckResult.from_dict(item) for item in value["checks"]),
            tuple(SourceBinding.from_dict(item) for item in value["evidence"]),
            tuple(value["reason_codes"]),
            str(value["outcome_sha256"]),
            str(value["schema_version"]),
        )


@runtime_checkable
class DomainPack(Protocol):
    """Minimal plug-in protocol; implementations own domain semantics, not promotion."""

    @property
    def descriptor(self) -> DomainPackDescriptor: ...

    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: Mapping[str, StageOutcome],
    ) -> StageOutcome: ...

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> GateOutcome: ...


def _error_check(label: str) -> CheckResult:
    return CheckResult.create("domain_pack_execution", False, {"boundary": label})


def run_stage(
    pack: DomainPack,
    artifact: CandidateArtifact,
    stage_id: str,
    prior_outcomes: Mapping[str, StageOutcome] | None = None,
) -> StageOutcome:
    """Evaluate a stage and convert pack exceptions/malformed results to an error outcome."""

    artifact.validate()
    descriptor = pack.descriptor
    if artifact.provenance.domain_pack != descriptor.ref:
        raise DomainPackViolation("candidate is not bound to this domain pack descriptor")
    stage = descriptor.stage(stage_id)
    if artifact.kind not in stage.allowed_kinds:
        raise DomainPackViolation("candidate kind is forbidden by stage")
    prior = dict(prior_outcomes or {})
    missing = [item for item in stage.prerequisites if item not in prior]
    extra = sorted(set(prior) - set(stage.prerequisites))
    invalid = [
        item
        for item in stage.prerequisites
        if item in prior
        and (
            not isinstance(prior[item], StageOutcome)
            or prior[item].stage_id != item
            or prior[item].status is not OutcomeStatus.PASS
            or prior[item].artifact != artifact.ref
        )
    ]
    if missing or extra or invalid:
        return StageOutcome.create(
            stage_id,
            artifact.ref,
            OutcomeStatus.BLOCK,
            [_error_check("prerequisite_outcomes_incomplete")],
            reason_codes=["prerequisite_outcomes_incomplete"],
        )
    try:
        outcome = pack.evaluate_stage(artifact, stage, prior)
        if not isinstance(outcome, StageOutcome):
            raise DomainPackViolation("domain pack returned a non-stage outcome")
        if outcome.stage_id != stage_id or outcome.artifact != artifact.ref:
            raise DomainPackViolation("domain pack returned an unbound stage outcome")
        return outcome
    except Exception as error:  # noqa: BLE001 - untrusted domain plug-in boundary
        return StageOutcome.create(
            stage_id,
            artifact.ref,
            OutcomeStatus.ERROR,
            [_error_check(type(error).__name__)],
            reason_codes=["domain_pack_error"],
        )


def run_gate(
    pack: DomainPack,
    artifact: CandidateArtifact,
    gate_id: str,
    stage_outcomes: Mapping[str, StageOutcome],
) -> GateOutcome:
    """Evaluate a gate only after exact passing required-stage outcomes exist."""

    artifact.validate()
    descriptor = pack.descriptor
    if artifact.provenance.domain_pack != descriptor.ref:
        raise DomainPackViolation("candidate is not bound to this domain pack descriptor")
    gate = descriptor.gate(gate_id)
    if artifact.kind not in gate.allowed_kinds:
        raise DomainPackViolation("candidate kind is forbidden by gate")
    provided = dict(stage_outcomes)
    valid_refs = tuple(
        outcome.ref
        for stage_id, outcome in sorted(provided.items())
        if isinstance(outcome, StageOutcome)
        and stage_id == outcome.stage_id
        and outcome.artifact == artifact.ref
    )
    if set(provided) != set(gate.required_stages) or any(
        not isinstance(provided[stage_id], StageOutcome)
        or provided[stage_id].status is not OutcomeStatus.PASS
        or provided[stage_id].artifact != artifact.ref
        or provided[stage_id].stage_id != stage_id
        for stage_id in set(provided) & set(gate.required_stages)
    ):
        return GateOutcome.create(
            gate_id,
            artifact.ref,
            OutcomeStatus.BLOCK,
            valid_refs,
            [_error_check("required_stage_outcomes_incomplete")],
            reason_codes=["required_stage_outcomes_incomplete"],
        )
    expected_refs = tuple(provided[item].ref for item in sorted(gate.required_stages))
    try:
        outcome = pack.evaluate_gate(artifact, gate, provided)
        if not isinstance(outcome, GateOutcome):
            raise DomainPackViolation("domain pack returned a non-gate outcome")
        if (
            outcome.gate_id != gate_id
            or outcome.artifact != artifact.ref
            or outcome.stage_outcomes != expected_refs
        ):
            raise DomainPackViolation("domain pack returned an unbound gate outcome")
        return outcome
    except Exception as error:  # noqa: BLE001 - untrusted domain plug-in boundary
        return GateOutcome.create(
            gate_id,
            artifact.ref,
            OutcomeStatus.ERROR,
            expected_refs,
            [_error_check(type(error).__name__)],
            reason_codes=["domain_pack_error"],
        )


@dataclass(frozen=True, slots=True)
class PromotionEntry:
    sequence: int
    gate_id: str
    from_stage: str | None
    to_stage: str
    gate_outcome_sha256: str
    prior_entry_sha256: str | None
    entry_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise SchemaViolation("promotion sequence must be a nonnegative integer")
        _identifier(self.gate_id, "promotion gate_id")
        if self.from_stage is not None:
            _identifier(self.from_stage, "promotion from_stage")
        _identifier(self.to_stage, "promotion to_stage")
        _hash(self.gate_outcome_sha256, "promotion gate outcome hash")
        if self.prior_entry_sha256 is not None:
            _hash(self.prior_entry_sha256, "prior promotion entry hash")
        _hash(self.entry_sha256, "promotion entry hash")
        if self.entry_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("promotion entry canonical hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "gate_id": self.gate_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "gate_outcome_sha256": self.gate_outcome_sha256,
            "prior_entry_sha256": self.prior_entry_sha256,
        }

    @classmethod
    def create(
        cls,
        sequence: int,
        gate_id: str,
        from_stage: str | None,
        to_stage: str,
        gate_outcome_sha256: str,
        prior_entry_sha256: str | None,
    ) -> PromotionEntry:
        body = {
            "sequence": sequence,
            "gate_id": gate_id,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "gate_outcome_sha256": gate_outcome_sha256,
            "prior_entry_sha256": prior_entry_sha256,
        }
        return cls(**body, entry_sha256=canonical_sha256(body))

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "entry_sha256": self.entry_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromotionEntry:
        _exact_keys(
            value,
            {
                "sequence",
                "gate_id",
                "from_stage",
                "to_stage",
                "gate_outcome_sha256",
                "prior_entry_sha256",
                "entry_sha256",
            },
            "promotion entry",
        )
        return cls(
            value["sequence"],
            str(value["gate_id"]),
            value["from_stage"],
            str(value["to_stage"]),
            str(value["gate_outcome_sha256"]),
            value["prior_entry_sha256"],
            str(value["entry_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PromotionLedger:
    artifact: ArtifactRef
    domain_pack: DomainPackRef
    entries: tuple[PromotionEntry, ...]
    ledger_sha256: str
    schema_version: str = "sigma-core-promotion-ledger-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sigma-core-promotion-ledger-1.0":
            raise SchemaViolation("promotion ledger schema_version changed")
        prior = None
        for index, entry in enumerate(self.entries):
            if entry.sequence != index or entry.prior_entry_sha256 != prior:
                raise SchemaViolation("promotion entry chain changed")
            if index and entry.from_stage != self.entries[index - 1].to_stage:
                raise SchemaViolation("promotion stage chain changed")
            prior = entry.entry_sha256
        _hash(self.ledger_sha256, "ledger_sha256")
        if self.ledger_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("promotion ledger canonical hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact": self.artifact.to_dict(),
            "domain_pack": self.domain_pack.to_dict(),
            "entries": [item.to_dict() for item in self.entries],
        }

    @classmethod
    def create(cls, artifact: CandidateArtifact) -> PromotionLedger:
        artifact.validate()
        body = {
            "schema_version": "sigma-core-promotion-ledger-1.0",
            "artifact": artifact.ref.to_dict(),
            "domain_pack": artifact.provenance.domain_pack.to_dict(),
            "entries": [],
        }
        return cls(artifact.ref, artifact.provenance.domain_pack, (), canonical_sha256(body))

    @property
    def current_stage(self) -> str | None:
        return self.entries[-1].to_stage if self.entries else None

    def promote(
        self,
        descriptor: DomainPackDescriptor,
        artifact: CandidateArtifact,
        gate_outcome: GateOutcome,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> PromotionLedger:
        """Append exactly one promotion or raise without changing the ledger."""

        if self.artifact != artifact.ref or self.domain_pack != descriptor.ref:
            raise PromotionDenied("ledger, candidate, and domain pack bindings disagree")
        try:
            artifact.validate()
        except SchemaViolation as error:
            raise PromotionDenied("candidate canonical identity is invalid") from error
        if artifact.provenance.domain_pack != descriptor.ref:
            raise PromotionDenied("candidate provenance is not bound to the domain pack")
        gate = descriptor.gate(gate_outcome.gate_id)
        if artifact.kind not in gate.allowed_kinds or gate.from_stage != self.current_stage:
            raise PromotionDenied("gate is not eligible from the current stage")
        if gate_outcome.status is not OutcomeStatus.PASS or gate_outcome.artifact != artifact.ref:
            raise PromotionDenied("gate did not pass for this candidate")
        if set(stage_outcomes) != set(gate.required_stages):
            raise PromotionDenied("required stage outcomes are incomplete")
        expected_refs = []
        for stage_id in sorted(gate.required_stages):
            outcome = stage_outcomes[stage_id]
            if (
                not isinstance(outcome, StageOutcome)
                or outcome.stage_id != stage_id
                or outcome.artifact != artifact.ref
                or outcome.status is not OutcomeStatus.PASS
            ):
                raise PromotionDenied("required stage outcome is not an exact pass")
            expected_refs.append(outcome.ref)
        if gate_outcome.stage_outcomes != tuple(expected_refs):
            raise PromotionDenied("gate is not bound to the supplied stage outcome hashes")
        entry = PromotionEntry.create(
            len(self.entries),
            gate.gate_id,
            gate.from_stage,
            gate.to_stage,
            gate_outcome.outcome_sha256,
            self.entries[-1].entry_sha256 if self.entries else None,
        )
        entries = (*self.entries, entry)
        body = {
            "schema_version": self.schema_version,
            "artifact": self.artifact.to_dict(),
            "domain_pack": self.domain_pack.to_dict(),
            "entries": [item.to_dict() for item in entries],
        }
        return PromotionLedger(
            self.artifact, self.domain_pack, entries, canonical_sha256(body), self.schema_version
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "ledger_sha256": self.ledger_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromotionLedger:
        _exact_keys(
            value,
            {"schema_version", "artifact", "domain_pack", "entries", "ledger_sha256"},
            "promotion ledger",
        )
        if not isinstance(value["entries"], list):
            raise SchemaViolation("promotion entries must be an array")
        return cls(
            ArtifactRef.from_dict(value["artifact"]),
            DomainPackRef.from_dict(value["domain_pack"]),
            tuple(PromotionEntry.from_dict(item) for item in value["entries"]),
            str(value["ledger_sha256"]),
            str(value["schema_version"]),
        )


__all__ = [
    "SCHEMA_VERSION",
    "ArtifactKind",
    "ArtifactRef",
    "CandidateArtifact",
    "CheckResult",
    "DomainPack",
    "DomainPackDescriptor",
    "DomainPackRef",
    "DomainPackViolation",
    "GateDefinition",
    "GateOutcome",
    "OutcomeRef",
    "OutcomeStatus",
    "PromotionDenied",
    "PromotionEntry",
    "PromotionLedger",
    "ProvenanceRecord",
    "SchemaViolation",
    "SigmaCoreError",
    "SourceBinding",
    "StageDefinition",
    "StageOutcome",
    "canonical_json_bytes",
    "canonical_sha256",
    "run_gate",
    "run_stage",
]
