"""Operational, data-driven runtime for extensible mathematical discovery.

The v1 protocol defines the safety boundary.  This module makes that boundary executable:
operators, invariants, proof tactics, and grammars arrive as strict typed data, pass independent
admission checks, and run through one bounded interpreter.  It also supplies the stronger
reachability, proof-plan, sealed-dataset, blind-benchmark, evidence-chain, and yield controls
needed before an open-ended campaign can make a serious claim.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from string import Formatter
from typing import Any

from . import declarative_discovery as D
from .sigma_core import canonical_sha256

EXTENSION_SCHEMA = "invariant-declarative-extension-candidate-2.0"
HEX_DIGITS = frozenset("0123456789abcdef")


class RuntimeProtocolError(ValueError):
    """A proposed extension or operational discovery receipt failed closed."""


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value)
    ):
        raise RuntimeProtocolError(f"{label} is not a portable identifier")
    return value


def _strict_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeProtocolError(f"{label} keys changed")


def _json_array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeProtocolError(f"{label} must be a JSON array")
    return value


def _text_array(value: Any, label: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise RuntimeProtocolError(f"{label} must be a JSON array of nonempty strings")
    return tuple(value)


def _sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise RuntimeProtocolError(f"{label} is not a lowercase SHA-256 digest")
    return value


def text_commitment(value: str) -> str:
    if not isinstance(value, str):
        raise RuntimeProtocolError("commitment preimage must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


class ProgramOpcode(str, Enum):
    FORMAT = "format"
    GRAMMAR_EXPAND = "grammar_expand"
    GOAL_TRANSITION = "goal_transition"


class VerifierBackend(str, Enum):
    SCHEMA = "schema"
    EXACT_ARITHMETIC = "exact_arithmetic"
    SMT = "smt"
    CAS = "cas"
    INTERVAL = "interval"
    LEAN = "lean"


class RuntimeStatus(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class GrammarProduction:
    nonterminal: str
    expansion: str

    def __post_init__(self) -> None:
        _identifier(self.nonterminal, "grammar nonterminal")
        if not isinstance(self.expansion, str) or not self.expansion.strip():
            raise RuntimeProtocolError("grammar expansion is empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GrammarProduction:
        _strict_keys(value, {"expansion", "nonterminal"}, "grammar production")
        return cls(value["nonterminal"], value["expansion"])

    def to_dict(self) -> dict[str, str]:
        return {"expansion": self.expansion, "nonterminal": self.nonterminal}


@dataclass(frozen=True, slots=True)
class DeclarativeProgram:
    opcode: ProgramOpcode
    template: str
    productions: tuple[GrammarProduction, ...]
    consumes: str
    produces: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.opcode, ProgramOpcode)
            or not isinstance(self.template, str)
            or not isinstance(self.consumes, str)
        ):
            raise RuntimeProtocolError("program opcode is not typed")
        if self.opcode is ProgramOpcode.FORMAT:
            if not self.template or self.productions or self.consumes or self.produces:
                raise RuntimeProtocolError("format program has incompatible fields")
        elif self.opcode is ProgramOpcode.GRAMMAR_EXPAND:
            if self.template or not self.productions or self.consumes or self.produces:
                raise RuntimeProtocolError("grammar program has incompatible fields")
            keys = [(item.nonterminal, item.expansion) for item in self.productions]
            if len(keys) != len(set(keys)):
                raise RuntimeProtocolError("grammar program has duplicate productions")
        elif self.opcode is ProgramOpcode.GOAL_TRANSITION:
            if self.template or self.productions or not self.consumes:
                raise RuntimeProtocolError("goal-transition program has incompatible fields")
            _identifier(self.consumes, "consumed goal kind")
            for item in self.produces:
                _identifier(item, "produced goal kind")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeclarativeProgram:
        _strict_keys(
            value,
            {"consumes", "opcode", "produces", "productions", "template"},
            "declarative program",
        )
        return cls(
            ProgramOpcode(value["opcode"]),
            value["template"],
            tuple(
                GrammarProduction.from_dict(item)
                for item in _json_array(value["productions"], "program productions")
            ),
            value["consumes"],
            _text_array(value["produces"], "program produces"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumes": self.consumes,
            "opcode": self.opcode.value,
            "produces": list(self.produces),
            "productions": [item.to_dict() for item in self.productions],
            "template": self.template,
        }

    def validate_arity(self, arity: int) -> None:
        if self.opcode is not ProgramOpcode.FORMAT:
            return
        fields = []
        for _, field_name, format_spec, conversion in Formatter().parse(self.template):
            if field_name is None:
                continue
            if format_spec or conversion or "." in field_name or "[" in field_name:
                raise RuntimeProtocolError("format program uses an unsafe placeholder")
            fields.append(field_name)
        expected = {f"arg{index}" for index in range(arity)}
        if set(fields) != expected:
            raise RuntimeProtocolError("format program placeholders do not match declared inputs")

    def execute(self, inputs: Sequence[str]) -> str | tuple[str, ...]:
        if any(not isinstance(item, str) or not item for item in inputs):
            raise RuntimeProtocolError("program inputs must be nonempty text")
        if self.opcode is ProgramOpcode.FORMAT:
            self.validate_arity(len(inputs))
            return self.template.format(**{f"arg{index}": item for index, item in enumerate(inputs)})
        if self.opcode is ProgramOpcode.GRAMMAR_EXPAND:
            if len(inputs) != 1:
                raise RuntimeProtocolError("grammar expansion consumes one nonterminal")
            outputs = tuple(
                item.expansion for item in self.productions if item.nonterminal == inputs[0]
            )
            if not outputs:
                raise RuntimeProtocolError("grammar has no production for the requested nonterminal")
            return outputs
        if len(inputs) != 1 or inputs[0] != self.consumes:
            raise RuntimeProtocolError("proof tactic does not consume this goal kind")
        return self.produces


@dataclass(frozen=True, slots=True)
class ExtensionTestVector:
    inputs: tuple[str, ...]
    expected: str | tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExtensionTestVector:
        _strict_keys(value, {"expected", "inputs"}, "extension test vector")
        expected = value["expected"]
        if isinstance(expected, list):
            expected = _text_array(expected, "extension expected outputs")
        if (
            not isinstance(expected, (str, tuple))
            or (isinstance(expected, str) and not expected)
        ):
            raise RuntimeProtocolError("extension test expectation is not typed")
        return cls(_text_array(value["inputs"], "extension test inputs"), expected)

    def to_dict(self) -> dict[str, Any]:
        expected: str | list[str]
        expected = list(self.expected) if isinstance(self.expected, tuple) else self.expected
        return {"expected": expected, "inputs": list(self.inputs)}


def _declaration_from_dict(value: Mapping[str, Any]) -> D.SearchDeclaration:
    _strict_keys(
        value,
        {"declaration_id", "input_types", "kind", "laws", "output_type", "symbols"},
        "search declaration",
    )
    symbols = []
    if not isinstance(value["symbols"], list):
        raise RuntimeProtocolError("typed symbols must be a JSON array")
    for item in value["symbols"]:
        _strict_keys(item, {"dimension", "name", "value_type"}, "typed symbol")
        dimension = item["dimension"]
        if not isinstance(dimension, list) or any(
            isinstance(part, bool) or not isinstance(part, int) for part in dimension
        ):
            raise RuntimeProtocolError("typed symbol dimension must be an integer JSON array")
        symbols.append(
            D.TypedSymbol(
                item["name"],
                D.ValueType(item["value_type"]),
                tuple(dimension),
            )
        )
    return D.SearchDeclaration(
        value["declaration_id"],
        D.DeclarationKind(value["kind"]),
        tuple(
            D.ValueType(item)
            for item in _text_array(value["input_types"], "declaration input types")
        ),
        D.ValueType(value["output_type"]),
        tuple(symbols),
        _text_array(value["laws"], "declaration laws", allow_empty=False),
    )


@dataclass(frozen=True, slots=True)
class ExtensionCandidate:
    candidate_id: str
    proposer_id: str
    declaration: D.SearchDeclaration
    program: DeclarativeProgram
    creativity_family: D.CreativityOperator | None
    capabilities: tuple[str, ...]
    tests: tuple[ExtensionTestVector, ...]

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "extension candidate_id")
        _identifier(self.proposer_id, "extension proposer_id")
        if tuple(sorted(set(self.capabilities))) != self.capabilities or not self.capabilities:
            raise RuntimeProtocolError("extension capabilities must be nonempty, sorted, and unique")
        for item in self.capabilities:
            _identifier(item, "extension capability")
        kind = self.declaration.kind
        expected_opcode = {
            D.DeclarationKind.OPERATOR: ProgramOpcode.FORMAT,
            D.DeclarationKind.INVARIANT: ProgramOpcode.FORMAT,
            D.DeclarationKind.PROOF_TACTIC: ProgramOpcode.GOAL_TRANSITION,
            D.DeclarationKind.GRAMMAR: ProgramOpcode.GRAMMAR_EXPAND,
        }[kind]
        if self.program.opcode is not expected_opcode:
            raise RuntimeProtocolError("declaration kind and generic program opcode disagree")
        if kind is D.DeclarationKind.OPERATOR and self.creativity_family is None:
            raise RuntimeProtocolError("operator extension must name its creativity family")
        if kind is not D.DeclarationKind.OPERATOR and self.creativity_family is not None:
            raise RuntimeProtocolError("only operator extensions name a creativity family")
        if kind in {D.DeclarationKind.OPERATOR, D.DeclarationKind.INVARIANT}:
            self.program.validate_arity(len(self.declaration.input_types))
        if kind is D.DeclarationKind.PROOF_TACTIC and (
            self.declaration.input_types != (D.ValueType.PROOF_STATE,)
            or self.declaration.output_type is not D.ValueType.PROOF_STATE
        ):
            raise RuntimeProtocolError("proof-tactic extension must transition proof states")
        if not self.tests:
            raise RuntimeProtocolError("extension admission requires replay tests")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExtensionCandidate:
        _strict_keys(
            value,
            {
                "candidate_id",
                "capabilities",
                "creativity_family",
                "declaration",
                "program",
                "proposer_id",
                "schema_version",
                "tests",
            },
            "extension candidate",
        )
        if value["schema_version"] != EXTENSION_SCHEMA:
            raise RuntimeProtocolError("extension candidate schema changed")
        family = value["creativity_family"]
        return cls(
            value["candidate_id"],
            value["proposer_id"],
            _declaration_from_dict(value["declaration"]),
            DeclarativeProgram.from_dict(value["program"]),
            D.CreativityOperator(family) if family is not None else None,
            _text_array(value["capabilities"], "extension capabilities", allow_empty=False),
            tuple(
                ExtensionTestVector.from_dict(item)
                for item in _json_array(value["tests"], "extension tests")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "capabilities": list(self.capabilities),
            "creativity_family": self.creativity_family.value if self.creativity_family else None,
            "declaration": self.declaration.to_dict(),
            "program": self.program.to_dict(),
            "proposer_id": self.proposer_id,
            "schema_version": EXTENSION_SCHEMA,
            "tests": [item.to_dict() for item in self.tests],
        }

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class VerifierIdentity:
    verifier_id: str
    principal_id: str
    backend: VerifierBackend
    implementation_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.verifier_id, "verifier_id")
        _identifier(self.principal_id, "verifier principal_id")
        if not isinstance(self.backend, VerifierBackend):
            raise RuntimeProtocolError("verifier backend is not typed")
        _sha256(self.implementation_sha256, "verifier implementation_sha256")


@dataclass(frozen=True, slots=True)
class ExtensionVerification:
    candidate_id: str
    candidate_sha256: str
    identity: VerifierIdentity
    status: RuntimeStatus
    obligations: tuple[str, ...]
    evidence_sha256: str
    blocker: D.TypedBlocker | None = None

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "verified extension candidate_id")
        _sha256(self.candidate_sha256, "verified candidate_sha256")
        _sha256(self.evidence_sha256, "extension evidence_sha256")
        if not self.obligations or tuple(sorted(set(self.obligations))) != self.obligations:
            raise RuntimeProtocolError("verified obligations must be nonempty, sorted, and unique")
        if self.status is RuntimeStatus.VERIFIED and self.blocker is not None:
            raise RuntimeProtocolError("verified extension retained a blocker")
        if self.status is not RuntimeStatus.VERIFIED and self.blocker is None:
            raise RuntimeProtocolError("rejected extension requires a typed blocker")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.identity.backend.value,
            "blocker": self.blocker.to_dict() if self.blocker else None,
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "evidence_sha256": self.evidence_sha256,
            "implementation_sha256": self.identity.implementation_sha256,
            "obligations": list(self.obligations),
            "principal_id": self.identity.principal_id,
            "status": self.status.value,
            "verifier_id": self.identity.verifier_id,
        }


ExtensionVerifier = Callable[[ExtensionCandidate, VerifierIdentity], ExtensionVerification]


class ExtensionVerifierRegistry:
    """Independent verifier identities are registered outside extension proposals."""

    def __init__(self) -> None:
        self._verifiers: dict[str, tuple[VerifierIdentity, ExtensionVerifier]] = {}

    def register(self, identity: VerifierIdentity, verifier: ExtensionVerifier) -> None:
        if (
            identity.verifier_id in self._verifiers
            or not callable(verifier)
            or any(existing is verifier for _, existing in self._verifiers.values())
        ):
            raise RuntimeProtocolError("extension verifier is duplicate or not callable")
        self._verifiers[identity.verifier_id] = identity, verifier

    def verify_all(self, candidate: ExtensionCandidate) -> tuple[ExtensionVerification, ...]:
        rows = []
        for identity, verifier in sorted(
            self._verifiers.values(), key=lambda item: item[0].verifier_id
        ):
            if identity.principal_id == candidate.proposer_id:
                raise RuntimeProtocolError("extension proposer cannot act as its own verifier")
            record = verifier(candidate, identity)
            if (
                record.candidate_id != candidate.candidate_id
                or record.candidate_sha256 != candidate.content_sha256
                or record.identity != identity
            ):
                raise RuntimeProtocolError("extension verifier changed bound provenance")
            rows.append(record)
        if not rows:
            raise RuntimeProtocolError("extension candidate has no independent verifier")
        return tuple(rows)


def structural_extension_verifier(
    candidate: ExtensionCandidate, identity: VerifierIdentity
) -> ExtensionVerification:
    """First verifier: re-check the strict schema, types, and bounded opcode contract."""

    replay = ExtensionCandidate.from_dict(candidate.to_dict())
    evidence = canonical_sha256(
        {
            "candidate": replay.content_sha256,
            "kind": replay.declaration.kind.value,
            "opcode": replay.program.opcode.value,
            "types": [item.value for item in replay.declaration.input_types],
        }
    )
    return ExtensionVerification(
        candidate.candidate_id,
        candidate.content_sha256,
        identity,
        RuntimeStatus.VERIFIED,
        ("bounded_opcode", "strict_schema", "type_transition"),
        evidence,
    )


def replay_extension_verifier(
    candidate: ExtensionCandidate, identity: VerifierIdentity
) -> ExtensionVerification:
    """Second verifier: independently execute every declared admission vector."""

    actual = []
    for vector in candidate.tests:
        output = candidate.program.execute(vector.inputs)
        if output != vector.expected:
            blocker = D.TypedBlocker(
                f"blocker.{candidate.candidate_id}.replay",
                D.BlockerKind.COUNTEREXAMPLE,
                candidate.declaration.output_type,
                Fraction(1),
                json.dumps(
                    {"actual": output, "expected": vector.expected, "inputs": vector.inputs},
                    sort_keys=True,
                ),
                D.CreativityOperator.COUNTEREXAMPLE_REPAIR,
            )
            return ExtensionVerification(
                candidate.candidate_id,
                candidate.content_sha256,
                identity,
                RuntimeStatus.REJECTED,
                ("admission_vectors",),
                canonical_sha256(blocker.to_dict()),
                blocker,
            )
        actual.append(output)
    return ExtensionVerification(
        candidate.candidate_id,
        candidate.content_sha256,
        identity,
        RuntimeStatus.VERIFIED,
        ("admission_vectors", "deterministic_replay"),
        canonical_sha256(actual),
    )


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    required_backends: tuple[VerifierBackend, ...]
    minimum_independent_principals: int = 2

    def __post_init__(self) -> None:
        if (
            not self.required_backends
            or len(set(self.required_backends)) != len(self.required_backends)
            or self.minimum_independent_principals < 2
        ):
            raise RuntimeProtocolError("extension admission policy is too weak")


@dataclass(frozen=True, slots=True)
class AdmittedExtension:
    candidate: ExtensionCandidate
    verification_sha256: str

    def execute(self, inputs: Sequence[str]) -> str | tuple[str, ...]:
        return self.candidate.program.execute(inputs)

    def emit_proposal(
        self, inputs: Sequence[D.Proposal], *, parent_ids: Sequence[str], nonce: str
    ) -> D.Proposal:
        if self.candidate.declaration.kind not in {
            D.DeclarationKind.OPERATOR,
            D.DeclarationKind.INVARIANT,
        }:
            raise RuntimeProtocolError("this admitted extension does not emit formula proposals")
        if tuple(item.value_type for item in inputs) != self.candidate.declaration.input_types:
            raise RuntimeProtocolError("extension proposal inputs violate the admitted types")
        _identifier(nonce, "extension execution nonce")
        representation = self.execute(tuple(item.representation for item in inputs))
        if not isinstance(representation, str):
            raise RuntimeProtocolError("formula-producing extension returned a non-formula")
        identity = canonical_sha256(
            {
                "admission": self.verification_sha256,
                "declaration": self.candidate.declaration.declaration_id,
                "inputs": [item.proposal_id for item in inputs],
                "nonce": nonce,
                "representation": representation,
            }
        )
        return D.Proposal(
            f"proposal-{identity[:24]}",
            self.candidate.declaration.declaration_id,
            self.candidate.creativity_family,
            self.candidate.declaration.output_type,
            representation,
            tuple(parent_ids),
            tuple(sorted({assumption for item in inputs for assumption in item.assumptions})),
        )

    def as_tactic(self) -> ProofTacticSpec:
        if self.candidate.declaration.kind is not D.DeclarationKind.PROOF_TACTIC:
            raise RuntimeProtocolError("this admitted extension is not a proof tactic")
        return ProofTacticSpec(
            self.candidate.declaration.declaration_id,
            self.candidate.program.consumes,
            self.candidate.program.produces,
            (),
            tuple(
                item.removeprefix("adds_invariant.")
                for item in self.candidate.capabilities
                if item.startswith("adds_invariant.")
            ),
            tuple(
                item.removeprefix("induction.")
                for item in self.candidate.capabilities
                if item.startswith("induction.")
            ),
            next(
                (
                    item.removeprefix("normal_form.")
                    for item in self.candidate.capabilities
                    if item.startswith("normal_form.")
                ),
                "unchanged",
            ),
            next(
                (
                    item.removeprefix("representation.")
                    for item in self.candidate.capabilities
                    if item.startswith("representation.")
                ),
                "native",
            ),
            next(
                (
                    item.removeprefix("mechanism.")
                    for item in self.candidate.capabilities
                    if item.startswith("mechanism.")
                ),
                self.candidate.declaration.declaration_id,
            ),
        )


class ExtensionAdmissionRegistry:
    def __init__(self, policy: AdmissionPolicy) -> None:
        self.policy = policy
        self._admitted: dict[str, AdmittedExtension] = {}

    def admit(
        self, candidate: ExtensionCandidate, records: Sequence[ExtensionVerification]
    ) -> AdmittedExtension:
        if candidate.declaration.declaration_id in self._admitted:
            raise RuntimeProtocolError("extension declaration is already admitted")
        matching = tuple(item for item in records if item.candidate_sha256 == candidate.content_sha256)
        if len(matching) != len(records) or any(
            item.status is not RuntimeStatus.VERIFIED for item in matching
        ):
            raise RuntimeProtocolError("extension admission contains rejection or changed candidate")
        principals = {item.identity.principal_id for item in matching}
        backends = {item.identity.backend for item in matching}
        implementations = {item.identity.implementation_sha256 for item in matching}
        if candidate.proposer_id in principals:
            raise RuntimeProtocolError("extension proposer appeared among admission verifiers")
        if (
            len(principals) < self.policy.minimum_independent_principals
            or len(implementations) < self.policy.minimum_independent_principals
            or not set(self.policy.required_backends) <= backends
        ):
            raise RuntimeProtocolError("extension did not meet independent admission policy")
        verification_sha256 = canonical_sha256([item.to_dict() for item in matching])
        admitted = AdmittedExtension(candidate, verification_sha256)
        self._admitted[candidate.declaration.declaration_id] = admitted
        return admitted

    def get(self, declaration_id: str) -> AdmittedExtension:
        try:
            return self._admitted[declaration_id]
        except KeyError as error:
            raise RuntimeProtocolError("extension declaration has not been admitted") from error

    @property
    def admitted_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._admitted))


@dataclass(frozen=True, slots=True)
class ProposedArtifact:
    proposal: D.Proposal
    proposer_principal_id: str

    def __post_init__(self) -> None:
        _identifier(self.proposer_principal_id, "proposal principal_id")


@dataclass(frozen=True, slots=True)
class BackendDecision:
    identity: VerifierIdentity
    record: D.VerificationRecord
    evidence_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.evidence_sha256, "backend evidence_sha256")


ResultVerifier = Callable[[D.Proposal, VerifierIdentity], BackendDecision]


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    required_backends: tuple[VerifierBackend, ...]
    minimum_independent_principals: int

    def __post_init__(self) -> None:
        if not self.required_backends or self.minimum_independent_principals < 2:
            raise RuntimeProtocolError("result decision policy is too weak")


@dataclass(frozen=True, slots=True)
class DecisionBundle:
    proposal_id: str
    decisions: tuple[BackendDecision, ...]
    bundle_sha256: str

    @property
    def verified(self) -> bool:
        return bool(self.decisions) and all(
            item.record.status is D.VerificationStatus.VERIFIED for item in self.decisions
        )


class IndependentResultVerifierRegistry:
    """Require distinct principals/backends before admitting a mathematical result."""

    def __init__(self, policy: DecisionPolicy) -> None:
        self.policy = policy
        self._verifiers: dict[str, tuple[VerifierIdentity, ResultVerifier]] = {}

    def register(self, identity: VerifierIdentity, verifier: ResultVerifier) -> None:
        if identity.verifier_id in self._verifiers or not callable(verifier):
            raise RuntimeProtocolError("result verifier is duplicate or not callable")
        self._verifiers[identity.verifier_id] = identity, verifier

    def decide(self, artifact: ProposedArtifact) -> DecisionBundle:
        decisions = []
        for identity, verifier in sorted(
            self._verifiers.values(), key=lambda item: item[0].verifier_id
        ):
            if identity.principal_id == artifact.proposer_principal_id:
                raise RuntimeProtocolError("proposal principal cannot verify its own result")
            decision = verifier(artifact.proposal, identity)
            if (
                decision.identity != identity
                or decision.record.proposal_id != artifact.proposal.proposal_id
                or decision.record.verifier_id != identity.verifier_id
            ):
                raise RuntimeProtocolError("result verifier changed bound provenance")
            decisions.append(decision)
        principals = {item.identity.principal_id for item in decisions}
        backends = {item.identity.backend for item in decisions}
        if len(principals) < self.policy.minimum_independent_principals or not set(
            self.policy.required_backends
        ) <= backends:
            raise RuntimeProtocolError("result lacks the required independent verifier quorum")
        body = [
            {
                "backend": item.identity.backend.value,
                "evidence_sha256": item.evidence_sha256,
                "record": item.record.to_dict(),
                "verifier": item.identity.verifier_id,
            }
            for item in decisions
        ]
        return DecisionBundle(artifact.proposal.proposal_id, tuple(decisions), canonical_sha256(body))


@dataclass(frozen=True, slots=True)
class SearchTransition:
    transition_id: str
    input_type: D.ValueType
    output_type: D.ValueType
    provides: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.transition_id, "search transition_id")
        if tuple(sorted(set(self.provides))) != self.provides:
            raise RuntimeProtocolError("transition features must be sorted and unique")


@dataclass(frozen=True, slots=True)
class TargetContract:
    target_id: str
    target_type: D.ValueType
    required_features: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.target_id, "target contract id")
        if not self.required_features or tuple(sorted(set(self.required_features))) != self.required_features:
            raise RuntimeProtocolError("target contract features must be nonempty, sorted, and unique")


@dataclass(frozen=True, slots=True)
class ExpressibilityCertificate:
    initial_type: D.ValueType
    target: TargetContract
    transition_path: tuple[str, ...]
    covered_features: tuple[str, ...]
    grammar_sha256: str
    witness_proposal_id: str
    witness_verification_sha256: str

    def validate(self, transitions: Sequence[SearchTransition]) -> None:
        _identifier(self.witness_proposal_id, "expressibility witness proposal_id")
        _sha256(self.grammar_sha256, "expressibility grammar_sha256")
        _sha256(self.witness_verification_sha256, "witness verification_sha256")
        by_id = {item.transition_id: item for item in transitions}
        current = self.initial_type
        features: set[str] = set()
        for transition_id in self.transition_path:
            transition = by_id.get(transition_id)
            if transition is None or transition.input_type is not current:
                raise RuntimeProtocolError("expressibility path contains an invalid transition")
            current = transition.output_type
            features.update(transition.provides)
        if current is not self.target.target_type:
            raise RuntimeProtocolError("expressibility path reaches the wrong value type")
        if tuple(sorted(features)) != self.covered_features:
            raise RuntimeProtocolError("expressibility feature coverage changed")
        if not set(self.target.required_features) <= features:
            raise RuntimeProtocolError("target class is not expressible in the declared grammar")
        expected_grammar = canonical_sha256(
            [
                {
                    "id": item.transition_id,
                    "input": item.input_type.value,
                    "output": item.output_type.value,
                    "provides": list(item.provides),
                }
                for item in sorted(transitions, key=lambda row: row.transition_id)
            ]
        )
        if self.grammar_sha256 != expected_grammar:
            raise RuntimeProtocolError("expressibility certificate binds a different grammar")


def prove_expressibility(
    initial_type: D.ValueType,
    target: TargetContract,
    transitions: Sequence[SearchTransition],
    *,
    witness_proposal_id: str,
    witness_verification_sha256: str,
) -> ExpressibilityCertificate:
    ordered = tuple(sorted(transitions, key=lambda item: item.transition_id))
    grammar_sha256 = canonical_sha256(
        [
            {
                "id": item.transition_id,
                "input": item.input_type.value,
                "output": item.output_type.value,
                "provides": list(item.provides),
            }
            for item in ordered
        ]
    )
    start = (initial_type, frozenset())
    queue: deque[tuple[D.ValueType, frozenset[str], tuple[str, ...]]] = deque(
        [(initial_type, frozenset(), ())]
    )
    seen = {start}
    while queue:
        current, features, path = queue.popleft()
        if current is target.target_type and set(target.required_features) <= features:
            result = ExpressibilityCertificate(
                initial_type,
                target,
                path,
                tuple(sorted(features)),
                grammar_sha256,
                witness_proposal_id,
                witness_verification_sha256,
            )
            result.validate(ordered)
            return result
        for transition in ordered:
            if transition.input_type is not current:
                continue
            next_features = features | frozenset(transition.provides)
            state = (transition.output_type, next_features)
            if state in seen:
                continue
            seen.add(state)
            queue.append((transition.output_type, next_features, (*path, transition.transition_id)))
    raise RuntimeProtocolError("target class is not expressible in the declared grammar")


@dataclass(frozen=True, slots=True)
class QualifiedNegative:
    target_id: str
    explored_proposals: int
    search_exhaustion_sha256: str
    expressibility: ExpressibilityCertificate
    status: str = "REACHABILITY_QUALIFIED_NEGATIVE"


def publish_qualified_negative(
    explored_proposals: int,
    search_exhaustion_sha256: str,
    certificate: ExpressibilityCertificate,
    transitions: Sequence[SearchTransition],
) -> QualifiedNegative:
    if explored_proposals < 1:
        raise RuntimeProtocolError("negative result requires a nonempty search")
    _sha256(search_exhaustion_sha256, "search exhaustion_sha256")
    certificate.validate(transitions)
    return QualifiedNegative(
        certificate.target.target_id,
        explored_proposals,
        search_exhaustion_sha256,
        certificate,
    )


@dataclass(frozen=True, slots=True)
class ProofGoal:
    goal_kind: str
    invariants: tuple[str, ...] = ()
    induction_variables: tuple[str, ...] = ()
    normal_form: str = "raw"
    representation: str = "native"

    def __post_init__(self) -> None:
        _identifier(self.goal_kind, "proof goal kind")
        _identifier(self.normal_form, "proof normal form")
        _identifier(self.representation, "proof representation")
        for values, label in (
            (self.invariants, "proof invariants"),
            (self.induction_variables, "proof induction variables"),
        ):
            if tuple(sorted(set(values))) != values:
                raise RuntimeProtocolError(f"{label} must be sorted and unique")


@dataclass(frozen=True, slots=True)
class ProofTacticSpec:
    tactic_id: str
    consumes: str
    produces: tuple[str, ...]
    requires_invariants: tuple[str, ...]
    adds_invariants: tuple[str, ...]
    introduces_induction: tuple[str, ...]
    normal_form: str
    representation: str
    mechanism: str

    def __post_init__(self) -> None:
        _identifier(self.tactic_id, "proof tactic_id")
        _identifier(self.consumes, "proof tactic consumed goal")
        _identifier(self.normal_form, "proof tactic normal form")
        _identifier(self.representation, "proof tactic representation")
        _identifier(self.mechanism, "proof tactic mechanism")
        for item in self.produces:
            _identifier(item, "proof tactic produced goal")


@dataclass(frozen=True, slots=True)
class OperationalProofPlan:
    proposal_id: str
    tactic_ids: tuple[str, ...]
    mechanisms: tuple[str, ...]
    closed: bool
    remaining_goals: tuple[ProofGoal, ...]
    blockers: tuple[D.TypedBlocker, ...]


def search_operational_proof_plan(
    proposal_id: str,
    initial_goals: Sequence[ProofGoal],
    tactics: Sequence[ProofTacticSpec],
    *,
    max_steps: int = 12,
) -> OperationalProofPlan:
    """Search lemma shape, invariants, induction choices, normal forms, and representations."""

    _identifier(proposal_id, "operational proof proposal_id")
    start = tuple(initial_goals)
    queue: deque[tuple[tuple[ProofGoal, ...], tuple[str, ...], tuple[str, ...]]] = deque(
        [(start, (), ())]
    )
    seen = {start}
    ordered = tuple(sorted(tactics, key=lambda item: item.tactic_id))
    best = start
    while queue:
        goals, plan, mechanisms = queue.popleft()
        if not goals:
            return OperationalProofPlan(
                proposal_id,
                plan,
                tuple(sorted(set(mechanisms))),
                True,
                (),
                (),
            )
        if len(goals) < len(best):
            best = goals
        if len(plan) >= max_steps:
            continue
        goal = goals[0]
        for tactic in ordered:
            if tactic.consumes != goal.goal_kind or not set(tactic.requires_invariants) <= set(
                goal.invariants
            ):
                continue
            invariants = tuple(sorted(set(goal.invariants) | set(tactic.adds_invariants)))
            induction = tuple(
                sorted(set(goal.induction_variables) | set(tactic.introduces_induction))
            )
            children = tuple(
                ProofGoal(
                    item,
                    invariants,
                    induction,
                    tactic.normal_form,
                    tactic.representation,
                )
                for item in tactic.produces
            )
            next_goals = (*children, *goals[1:])
            if next_goals in seen:
                continue
            seen.add(next_goals)
            queue.append(
                (next_goals, (*plan, tactic.tactic_id), (*mechanisms, tactic.mechanism))
            )
    distance = Fraction(len(best), max(1, len(start) + max_steps))
    blocker = D.TypedBlocker(
        f"blocker.{proposal_id}.proof-plan",
        D.BlockerKind.PROOF_OBLIGATION,
        D.ValueType.PROOF_STATE,
        distance,
        json.dumps(
            [
                {
                    "goal_kind": item.goal_kind,
                    "induction_variables": item.induction_variables,
                    "invariants": item.invariants,
                    "normal_form": item.normal_form,
                    "representation": item.representation,
                }
                for item in best
            ],
            sort_keys=True,
        ),
        D.CreativityOperator.COUNTEREXAMPLE_REPAIR,
    )
    return OperationalProofPlan(proposal_id, (), (), False, best, (blocker,))


class DatasetStageV2(str, Enum):
    SHAPE_AUDIT = "shape_audit"
    UNIT_NORMALIZATION = "unit_normalization"
    DIMENSIONLESS_GROUPS = "dimensionless_groups"
    INVARIANT_COORDINATES = "invariant_coordinates"
    TRAIN_HOLDOUT_SPLIT = "train_holdout_split"
    SYMBOLIC_LAW_FIT = "symbolic_law_fit"
    STRUCTURED_RESIDUAL_CHANNELS = "structured_residual_channels"
    MECHANISM_FALSIFIER = "mechanism_falsifier"
    INDEPENDENT_REPRODUCTION = "independent_reproduction"
    HELDOUT_TEST = "heldout_test"


DATASET_STAGES_V2 = tuple(DatasetStageV2)


@dataclass(frozen=True, slots=True)
class DatasetEvidence:
    stage: DatasetStageV2
    input_sha256: str
    output_sha256: str
    artifacts: tuple[str, ...]
    metric: Fraction
    passed: bool
    blocker: D.TypedBlocker | None
    heldout_reveal_sha256: str | None


class OperationalDatasetPipeline:
    """Hash-chained explanation stages with the committed holdout opened only at the end."""

    def __init__(self, dataset_sha256: str, heldout_commitment: str) -> None:
        self.dataset_sha256 = _sha256(dataset_sha256, "dataset_sha256")
        self.heldout_commitment = _sha256(heldout_commitment, "heldout commitment")
        self.records: list[DatasetEvidence] = []

    def record(
        self,
        stage: DatasetStageV2,
        input_sha256: str,
        output_sha256: str,
        *,
        artifacts: Sequence[str],
        metric: Fraction,
        passed: bool,
        blocker: D.TypedBlocker | None = None,
        heldout_reveal: str | None = None,
    ) -> DatasetEvidence:
        if self.records and not self.records[-1].passed:
            raise RuntimeProtocolError("dataset pipeline cannot continue after a failed stage")
        expected_stage = (
            DATASET_STAGES_V2[len(self.records)]
            if len(self.records) < len(DATASET_STAGES_V2)
            else None
        )
        if stage is not expected_stage:
            raise RuntimeProtocolError("dataset explanation stage is missing or out of order")
        expected_input = self.records[-1].output_sha256 if self.records else self.dataset_sha256
        if input_sha256 != expected_input:
            raise RuntimeProtocolError("dataset explanation hash chain is broken")
        _sha256(output_sha256, "dataset stage output_sha256")
        if not isinstance(metric, Fraction) or metric < 0:
            raise RuntimeProtocolError("dataset stage metric must be an exact nonnegative fraction")
        if tuple(sorted(set(artifacts))) != tuple(artifacts) or not artifacts:
            raise RuntimeProtocolError("dataset stage artifacts must be nonempty, sorted, and unique")
        if passed == (blocker is not None):
            raise RuntimeProtocolError("failed dataset stage requires exactly one typed blocker")
        reveal_sha = None
        if stage is DatasetStageV2.HELDOUT_TEST:
            if heldout_reveal is None or text_commitment(heldout_reveal) != self.heldout_commitment:
                raise RuntimeProtocolError("heldout reveal does not open the sealed commitment")
            reveal_sha = text_commitment(heldout_reveal)
        elif heldout_reveal is not None:
            raise RuntimeProtocolError("heldout object was opened before the final test")
        row = DatasetEvidence(
            stage,
            input_sha256,
            output_sha256,
            tuple(artifacts),
            metric,
            passed,
            blocker,
            reveal_sha,
        )
        self.records.append(row)
        return row

    @property
    def completed(self) -> bool:
        return len(self.records) == len(DATASET_STAGES_V2) and all(
            item.passed for item in self.records
        )


class CapabilityLevelV2(int, Enum):
    SOLVED_VISIBLE = 1
    SOLVED_ANONYMOUS = 2
    SYNTHETIC_TARGET_SEALED = 3
    HISTORICAL_TARGET_SEALED = 4
    BOUNDED_UNKNOWN_DECIDABLE = 5
    OPEN_PROBLEM = 6


@dataclass(frozen=True, slots=True)
class BlindBenchmarkResult:
    level: CapabilityLevelV2
    benchmark_id: str
    proposer_principal_id: str
    target_commitment: str
    proposal_sha256: str
    proposal_sequence: int
    target_reveal: str
    reveal_sequence: int
    leakage_tokens: tuple[str, ...]
    verifier_backends: tuple[VerifierBackend, ...]
    passed: bool

    def __post_init__(self) -> None:
        _identifier(self.benchmark_id, "blind benchmark_id")
        _identifier(self.proposer_principal_id, "blind proposer principal_id")
        _sha256(self.proposal_sha256, "blind proposal_sha256")
        if self.proposal_sequence < 0 or self.reveal_sequence <= self.proposal_sequence:
            raise RuntimeProtocolError("blind target was not opened after proposal commitment")
        if self.level.value >= CapabilityLevelV2.SYNTHETIC_TARGET_SEALED.value:
            _sha256(self.target_commitment, "blind target commitment")
            if text_commitment(self.target_reveal) != self.target_commitment:
                raise RuntimeProtocolError("blind target reveal does not match its commitment")
            if self.leakage_tokens:
                raise RuntimeProtocolError("sealed blind benchmark leaked target tokens")
        if self.passed and not {
            VerifierBackend.EXACT_ARITHMETIC,
            VerifierBackend.LEAN,
        } & set(self.verifier_backends):
            raise RuntimeProtocolError("passed blind benchmark lacks an exact or kernel verifier")


class BlindCapabilityLadderV2:
    def __init__(self) -> None:
        self.results: list[BlindBenchmarkResult] = []

    def admit(self, result: BlindBenchmarkResult) -> None:
        if result.level.value != len(self.results) + 1:
            raise RuntimeProtocolError("blind capability levels cannot be skipped")
        if self.results and not self.results[-1].passed:
            raise RuntimeProtocolError("blind ladder cannot advance past a failed level")
        self.results.append(result)

    @property
    def highest_passed(self) -> int:
        return max((item.level.value for item in self.results if item.passed), default=0)

    @property
    def open_problem_spend_authorized(self) -> bool:
        return self.highest_passed >= CapabilityLevelV2.BOUNDED_UNKNOWN_DECIDABLE.value


class EvidenceStage(int, Enum):
    DECLARATION = 1
    TARGET_COMMITMENT = 2
    BLIND_PROPOSAL = 3
    HOLDOUT_SURVIVAL = 4
    INDEPENDENT_REPRODUCTION = 5
    EXACT_CERTIFICATE = 6
    KERNEL_PROOF = 7
    HUMAN_PRIOR_ART = 8
    RELEASE = 9


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    link_id: str
    stage: EvidenceStage
    artifact_sha256: str
    actor_id: str
    parent_ids: tuple[str, ...]
    evidence_kind: str

    def __post_init__(self) -> None:
        _identifier(self.link_id, "evidence link_id")
        _identifier(self.actor_id, "evidence actor_id")
        _identifier(self.evidence_kind, "evidence kind")
        _sha256(self.artifact_sha256, "evidence artifact_sha256")


class SeriousClaimChain:
    """Provenance DAG with a fail-closed release policy for serious claims."""

    def __init__(self) -> None:
        self.links: dict[str, EvidenceLink] = {}

    def add(self, link: EvidenceLink) -> None:
        if link.link_id in self.links:
            raise RuntimeProtocolError("duplicate serious-claim evidence link")
        if link.stage is EvidenceStage.DECLARATION:
            if link.parent_ids:
                raise RuntimeProtocolError("declaration evidence cannot have parents")
        elif not link.parent_ids:
            raise RuntimeProtocolError("non-declaration evidence requires parents")
        for parent_id in link.parent_ids:
            parent = self.links.get(parent_id)
            if parent is None or parent.stage.value >= link.stage.value:
                raise RuntimeProtocolError("evidence parent is absent or not earlier")
        self.links[link.link_id] = link

    def _ancestors(self, link_id: str) -> tuple[EvidenceLink, ...]:
        if link_id not in self.links:
            raise RuntimeProtocolError("release evidence link is absent")
        found: dict[str, EvidenceLink] = {}
        stack = [link_id]
        while stack:
            current = self.links[stack.pop()]
            if current.link_id in found:
                continue
            found[current.link_id] = current
            stack.extend(current.parent_ids)
        return tuple(found.values())

    def validate_release(self, release_id: str) -> str:
        release = self.links.get(release_id)
        if release is None or release.stage is not EvidenceStage.RELEASE:
            raise RuntimeProtocolError("serious claim has no release-stage evidence")
        ancestors = self._ancestors(release_id)
        stages = {item.stage for item in ancestors}
        required = {
            EvidenceStage.DECLARATION,
            EvidenceStage.TARGET_COMMITMENT,
            EvidenceStage.BLIND_PROPOSAL,
            EvidenceStage.HOLDOUT_SURVIVAL,
            EvidenceStage.INDEPENDENT_REPRODUCTION,
            EvidenceStage.HUMAN_PRIOR_ART,
            EvidenceStage.RELEASE,
        }
        if not required <= stages or not {
            EvidenceStage.EXACT_CERTIFICATE,
            EvidenceStage.KERNEL_PROOF,
        } & stages:
            raise RuntimeProtocolError("serious claim release chain is incomplete")
        proposer = next(item for item in ancestors if item.stage is EvidenceStage.BLIND_PROPOSAL)
        holdout = next(item for item in ancestors if item.stage is EvidenceStage.HOLDOUT_SURVIVAL)
        reproduction = next(
            item for item in ancestors if item.stage is EvidenceStage.INDEPENDENT_REPRODUCTION
        )
        prior_art = next(item for item in ancestors if item.stage is EvidenceStage.HUMAN_PRIOR_ART)
        if len({proposer.actor_id, holdout.actor_id, reproduction.actor_id}) < 3:
            raise RuntimeProtocolError("proposal, holdout, and reproduction actors are not independent")
        if prior_art.evidence_kind != "human_review":
            raise RuntimeProtocolError("prior-art gate was not a human review")
        return canonical_sha256(
            [
                {
                    "actor": item.actor_id,
                    "artifact": item.artifact_sha256,
                    "id": item.link_id,
                    "kind": item.evidence_kind,
                    "parents": list(item.parent_ids),
                    "stage": item.stage.name.lower(),
                }
                for item in sorted(ancestors, key=lambda row: row.link_id)
            ]
        )


@dataclass(frozen=True, slots=True)
class OperationalCreativeYield:
    proposals: int
    verified: int
    behavioral_niches: int
    unique_proof_mechanisms: int
    proof_plans_attempted: int
    proof_plans_closed: int
    counterexamples_tested: int
    counterexample_survivors: int
    holdout_baseline_loss: Fraction
    holdout_best_loss: Fraction
    positives: int
    gpu_milliseconds_construction: int
    gpu_milliseconds_refutation: int

    def __post_init__(self) -> None:
        counts = (
            self.proposals,
            self.verified,
            self.behavioral_niches,
            self.unique_proof_mechanisms,
            self.proof_plans_attempted,
            self.proof_plans_closed,
            self.counterexamples_tested,
            self.counterexample_survivors,
            self.positives,
            self.gpu_milliseconds_construction,
            self.gpu_milliseconds_refutation,
        )
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
            raise RuntimeProtocolError("operational creative-yield counts are invalid")
        if (
            self.verified > self.proposals
            or self.behavioral_niches > self.verified
            or self.proof_plans_closed > self.proof_plans_attempted
            or self.counterexample_survivors > self.counterexamples_tested
            or self.holdout_baseline_loss < 0
            or self.holdout_best_loss < 0
        ):
            raise RuntimeProtocolError("operational creative-yield values are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        total_gpu = self.gpu_milliseconds_construction + self.gpu_milliseconds_refutation

        def ratio(numerator: int | Fraction, denominator: int | Fraction) -> str:
            return _fraction(Fraction(numerator, denominator)) if denominator else "0/1"

        improvement = (
            (self.holdout_baseline_loss - self.holdout_best_loss)
            / self.holdout_baseline_loss
            if self.holdout_baseline_loss
            else Fraction(0)
        )
        return {
            "claims": {
                "novelty_established": False,
                "truth_established_by_metric": False,
            },
            "compute": {
                "construction_gpu_hours": ratio(
                    self.gpu_milliseconds_construction, 3_600_000
                ),
                "construction_to_refutation": ratio(
                    self.gpu_milliseconds_construction,
                    self.gpu_milliseconds_refutation,
                ),
                "gpu_hours_per_positive": ratio(total_gpu, 3_600_000 * self.positives),
                "refutation_gpu_hours": ratio(self.gpu_milliseconds_refutation, 3_600_000),
            },
            "counts": {
                "behavioral_niches": self.behavioral_niches,
                "counterexample_survivors": self.counterexample_survivors,
                "counterexamples_tested": self.counterexamples_tested,
                "positives": self.positives,
                "proof_plans_attempted": self.proof_plans_attempted,
                "proof_plans_closed": self.proof_plans_closed,
                "proposals": self.proposals,
                "unique_proof_mechanisms": self.unique_proof_mechanisms,
                "verified": self.verified,
            },
            "rates": {
                "counterexample_survival": ratio(
                    self.counterexample_survivors, self.counterexamples_tested
                ),
                "holdout_improvement": _fraction(improvement),
                "proof_completion": ratio(self.proof_plans_closed, self.proof_plans_attempted),
                "verification_yield": ratio(self.verified, self.proposals),
            },
        }
