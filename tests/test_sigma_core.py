from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    ArtifactRef,
    CandidateArtifact,
    CheckResult,
    DomainPack,
    DomainPackDescriptor,
    DomainPackRef,
    DomainPackViolation,
    GateDefinition,
    GateOutcome,
    OutcomeRef,
    OutcomeStatus,
    PromotionDenied,
    PromotionLedger,
    ProvenanceRecord,
    SchemaViolation,
    SourceBinding,
    StageDefinition,
    StageOutcome,
    canonical_json_bytes,
    canonical_sha256,
    run_gate,
    run_stage,
)

ROOT = Path(__file__).resolve().parents[1]
ZERO = "0" * 64
ONE = "1" * 64
KINDS = tuple(sorted(ArtifactKind, key=lambda item: item.value))


def descriptor() -> DomainPackDescriptor:
    return DomainPackDescriptor(
        pack_id="example.math",
        pack_version="1.0.0",
        supported_kinds=KINDS,
        stages=(
            StageDefinition("syntax", 0, KINDS),
            StageDefinition("verified", 1, KINDS, ("syntax",)),
        ),
        gates=(
            GateDefinition("accept_syntax", None, "syntax", KINDS, ("syntax",)),
            GateDefinition("admit_verified", "syntax", "verified", KINDS, ("syntax", "verified")),
        ),
    )


def provenance(desc: DomainPackDescriptor | None = None) -> ProvenanceRecord:
    selected = desc or descriptor()
    return ProvenanceRecord.create(
        selected.ref,
        {"budget": 7, "exact_ring": "Q"},
        inputs=(ArtifactRef("sig-parent", ZERO),),
        sources=(SourceBinding("generator", "packs/example.py", ONE),),
    )


def candidate(
    desc: DomainPackDescriptor | None = None,
    *,
    kind: ArtifactKind = ArtifactKind.FORMULA,
) -> CandidateArtifact:
    return CandidateArtifact.create(
        kind,
        "For every input, the declared relation holds.",
        {"operator": "equals", "arguments": ["left", "right"], "degree": 2},
        provenance(desc),
        assumptions=("finite input",),
        claims=("declared_relation",),
    )


def passed_stage(artifact: CandidateArtifact, stage_id: str) -> StageOutcome:
    return StageOutcome.create(
        stage_id,
        artifact.ref,
        OutcomeStatus.PASS,
        (CheckResult.create(f"{stage_id}_check", True, {"exact": True}),),
        evidence=(SourceBinding("certificate", f"evidence/{stage_id}.json", ZERO),),
    )


class ExamplePack:
    def __init__(self, desc: DomainPackDescriptor | None = None) -> None:
        self._descriptor = desc or descriptor()

    @property
    def descriptor(self) -> DomainPackDescriptor:
        return self._descriptor

    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: dict[str, StageOutcome],
    ) -> StageOutcome:
        del prior_outcomes
        return passed_stage(artifact, stage.stage_id)

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: dict[str, StageOutcome],
    ) -> GateOutcome:
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            OutcomeStatus.PASS,
            tuple(stage_outcomes[key].ref for key in sorted(stage_outcomes)),
            (CheckResult.create("promotion_contract", True, {"closed": True}),),
        )


class ExplodingPack(ExamplePack):
    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: dict[str, StageOutcome],
    ) -> StageOutcome:
        raise RuntimeError("untrusted plug-in detail")

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: dict[str, StageOutcome],
    ) -> GateOutcome:
        raise RuntimeError("untrusted plug-in detail")


def reseal_stage(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "outcome_sha256"}
    return {**body, "outcome_sha256": canonical_sha256(body)}


def reseal_gate(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "outcome_sha256"}
    return {**body, "outcome_sha256": canonical_sha256(body)}


def test_core_has_no_domain_imports_or_gravity_dependency() -> None:
    source = (ROOT / "src/sigma_theory_compiler/sigma_core.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert all(not name.startswith("sigma_theory_compiler") for name in imported)
    assert "gravity" not in source.lower()


def test_canonical_json_is_order_independent_utf8_and_exact() -> None:
    left = {"z": [1, True, None], "a": "λ"}
    right = {"a": "λ", "z": [1, True, None]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_bytes(left) == b'{"a":"\xce\xbb","z":[1,true,null]}'
    assert canonical_sha256(left) == canonical_sha256(right)


@pytest.mark.parametrize("bad", [1.0, float("nan"), float("inf")])
def test_canonical_json_rejects_all_floats(bad: float) -> None:
    with pytest.raises(SchemaViolation, match="floating value forbidden"):
        canonical_json_bytes({"bad": bad})


@pytest.mark.parametrize("bad", [{1: "x"}, {"x": {1, 2}}, {"x": b"bytes"}])
def test_canonical_json_rejects_non_json_values(bad: object) -> None:
    with pytest.raises(SchemaViolation):
        canonical_json_bytes(bad)


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_candidate_ir_covers_every_closed_world_kind(kind: ArtifactKind) -> None:
    artifact = candidate(kind=kind)
    replayed = CandidateArtifact.from_dict(json.loads(json.dumps(artifact.to_dict())))
    assert replayed == artifact
    assert artifact.kind is kind
    assert artifact.artifact_id == f"sig-{artifact.content_sha256[:24]}"


def test_candidate_creation_detaches_payload_and_sorts_sets() -> None:
    payload = {"items": ["a"]}
    artifact = CandidateArtifact.create(
        ArtifactKind.CONSTRUCTION,
        "Construct the declared object.",
        payload,
        provenance(),
        assumptions=("z premise", "a premise"),
        claims=("z_claim", "a_claim"),
    )
    payload["items"].append("tamper")
    assert artifact.representation == {"items": ["a"]}
    assert artifact.assumptions == ("a premise", "z premise")
    assert artifact.claims == ("a_claim", "z_claim")
    exported = artifact.to_dict()
    exported["representation"]["items"].append("export-tamper")
    assert artifact.representation == {"items": ["a"]}


def test_mutated_candidate_payload_is_rejected_at_execution_boundary() -> None:
    desc = descriptor()
    artifact = candidate(desc)
    artifact.representation["degree"] = 99  # type: ignore[index]
    with pytest.raises(SchemaViolation, match="canonical identity"):
        run_stage(ExamplePack(desc), artifact, "syntax")
    with pytest.raises(SchemaViolation, match="canonical identity"):
        PromotionLedger.create(artifact)


@pytest.mark.parametrize("field", ["statement", "representation", "content_sha256", "artifact_id"])
def test_candidate_tampering_fails_closed(field: str) -> None:
    value = candidate().to_dict()
    if field == "statement":
        value[field] = "A different statement."
    elif field == "representation":
        value[field]["degree"] = 3
    else:
        value[field] = ZERO
    with pytest.raises(SchemaViolation):
        CandidateArtifact.from_dict(value)


def test_candidate_unknown_field_and_unknown_kind_fail_closed() -> None:
    value = candidate().to_dict()
    value["promotion"] = "verified"
    with pytest.raises(SchemaViolation, match="keys changed"):
        CandidateArtifact.from_dict(value)
    value = candidate().to_dict()
    value["kind"] = "hypothesis"
    with pytest.raises(SchemaViolation, match="kind"):
        CandidateArtifact.from_dict(value)


@pytest.mark.parametrize(
    "path",
    ["/absolute/file.json", "../escape.json", "a/../escape.json", r"a\windows.json", "a/./b"],
)
def test_source_bindings_reject_nonportable_or_escaping_paths(path: str) -> None:
    with pytest.raises(SchemaViolation, match="path"):
        SourceBinding("evidence", path, ZERO)


def test_provenance_is_canonical_and_binding_complete() -> None:
    desc = descriptor()
    first = ProvenanceRecord.create(
        desc.ref,
        {"b": 2, "a": 1},
        inputs=(ArtifactRef("sig-z", ZERO), ArtifactRef("sig-a", ONE)),
        sources=(
            SourceBinding("test", "tests/test_pack.py", ZERO),
            SourceBinding("source", "packs/source.py", ONE),
        ),
    )
    second = ProvenanceRecord.create(
        desc.ref,
        {"a": 1, "b": 2},
        inputs=tuple(reversed(first.inputs)),
        sources=tuple(reversed(first.sources)),
    )
    assert first == second == ProvenanceRecord.from_dict(first.to_dict())
    assert first.inputs[0].artifact_id == "sig-a"
    assert first.sources[0].role == "source"


def test_provenance_rejects_duplicate_roles_and_unsorted_inputs() -> None:
    with pytest.raises(SchemaViolation, match="sorted"):
        ProvenanceRecord(
            descriptor().ref,
            ZERO,
            (ArtifactRef("sig-z", ZERO), ArtifactRef("sig-a", ONE)),
        )
    with pytest.raises(SchemaViolation, match="duplicates"):
        ProvenanceRecord(
            descriptor().ref,
            ZERO,
            sources=(
                SourceBinding("source", "a", ZERO),
                SourceBinding("source", "b", ONE),
            ),
        )


def test_descriptor_is_canonical_and_protocol_is_runtime_visible() -> None:
    desc = descriptor()
    assert isinstance(ExamplePack(desc), DomainPack)
    assert DomainPackDescriptor.from_dict(desc.to_dict()) == desc
    assert desc.ref == DomainPackRef(
        desc.pack_id, desc.pack_version, canonical_sha256(desc.to_dict())
    )
    assert desc.stage("verified").prerequisites == ("syntax",)
    assert desc.gate("accept_syntax").from_stage is None


@pytest.mark.parametrize("target", ["unknown", "kind", "ordinal", "prerequisite"])
def test_descriptor_tampering_fails_closed(target: str) -> None:
    value = descriptor().to_dict()
    if target == "unknown":
        value["domain_semantics"] = "trusted"
    elif target == "kind":
        value["supported_kinds"][0] = "unregistered_kind"
    elif target == "ordinal":
        value["stages"][1]["ordinal"] = 0
    else:
        value["stages"][0]["prerequisites"] = ["verified"]
    with pytest.raises(SchemaViolation):
        DomainPackDescriptor.from_dict(value)


def test_descriptor_rejects_bad_order_dependencies_and_gate_closure() -> None:
    with pytest.raises(SchemaViolation, match="consecutive"):
        DomainPackDescriptor(
            "bad.pack",
            "1",
            KINDS,
            (StageDefinition("syntax", 1, KINDS),),
            (),
        )
    with pytest.raises(SchemaViolation, match="earlier"):
        DomainPackDescriptor(
            "bad.pack",
            "1",
            KINDS,
            (StageDefinition("syntax", 0, KINDS, ("syntax",)),),
            (),
        )
    with pytest.raises(SchemaViolation, match="include to_stage"):
        GateDefinition("bad_gate", None, "syntax", KINDS, ())


def test_stage_and_gate_outcomes_round_trip_with_exact_hashes() -> None:
    artifact = candidate()
    syntax = passed_stage(artifact, "syntax")
    gate = GateOutcome.create(
        "accept_syntax",
        artifact.ref,
        OutcomeStatus.PASS,
        (syntax.ref,),
        (CheckResult.create("promotion_contract", True, {"ok": True}),),
    )
    assert StageOutcome.from_dict(syntax.to_dict()) == syntax
    assert GateOutcome.from_dict(gate.to_dict()) == gate
    assert syntax.ref == OutcomeRef("syntax", syntax.outcome_sha256)
    assert gate.ref == OutcomeRef("accept_syntax", gate.outcome_sha256)


@pytest.mark.parametrize("status", [OutcomeStatus.BLOCK, OutcomeStatus.REJECT, OutcomeStatus.ERROR])
def test_nonpass_outcomes_require_failed_check_and_reason(status: OutcomeStatus) -> None:
    artifact = candidate()
    with pytest.raises(SchemaViolation, match="requires reasons"):
        StageOutcome.create(
            "syntax",
            artifact.ref,
            status,
            (CheckResult.create("check", True, {"ok": True}),),
        )


def test_pass_outcomes_cannot_hide_failed_checks_or_reasons() -> None:
    artifact = candidate()
    with pytest.raises(SchemaViolation, match="pass outcome"):
        StageOutcome.create(
            "syntax",
            artifact.ref,
            OutcomeStatus.PASS,
            (CheckResult.create("check", False, {"ok": False}),),
        )
    with pytest.raises(SchemaViolation, match="pass outcome"):
        StageOutcome.create(
            "syntax",
            artifact.ref,
            OutcomeStatus.PASS,
            (CheckResult.create("check", True, {"ok": True}),),
            reason_codes=("unexpected_reason",),
        )


def test_stage_runner_enforces_prerequisites_and_pack_binding() -> None:
    desc = descriptor()
    pack = ExamplePack(desc)
    artifact = candidate(desc)
    syntax = run_stage(pack, artifact, "syntax")
    blocked = run_stage(pack, artifact, "verified")
    verified = run_stage(pack, artifact, "verified", {"syntax": syntax})
    assert syntax.status is OutcomeStatus.PASS
    assert blocked.status is OutcomeStatus.BLOCK
    assert blocked.reason_codes == ("prerequisite_outcomes_incomplete",)
    assert verified.status is OutcomeStatus.PASS
    extra = run_stage(pack, artifact, "verified", {"syntax": syntax, "unregistered": syntax})
    assert extra.status is OutcomeStatus.BLOCK
    other = DomainPackDescriptor(
        descriptor().pack_id,
        "2.0.0",
        descriptor().supported_kinds,
        descriptor().stages,
        descriptor().gates,
    )
    with pytest.raises(DomainPackViolation, match="not bound"):
        run_stage(ExamplePack(other), artifact, "syntax")


def test_pack_exceptions_and_malformed_outcomes_become_error_outcomes() -> None:
    desc = descriptor()
    artifact = candidate(desc)
    exploding = ExplodingPack(desc)
    stage_error = run_stage(exploding, artifact, "syntax")
    syntax = passed_stage(artifact, "syntax")
    gate_error = run_gate(exploding, artifact, "accept_syntax", {"syntax": syntax})
    assert stage_error.status is OutcomeStatus.ERROR
    assert gate_error.status is OutcomeStatus.ERROR
    assert stage_error.reason_codes == gate_error.reason_codes == ("domain_pack_error",)
    assert "untrusted" not in json.dumps(stage_error.to_dict())


def test_gate_runner_requires_exact_complete_passing_stage_map() -> None:
    desc = descriptor()
    pack = ExamplePack(desc)
    artifact = candidate(desc)
    syntax = passed_stage(artifact, "syntax")
    verified = passed_stage(artifact, "verified")
    blocked = run_gate(pack, artifact, "admit_verified", {"syntax": syntax})
    passed = run_gate(pack, artifact, "admit_verified", {"verified": verified, "syntax": syntax})
    assert blocked.status is OutcomeStatus.BLOCK
    assert passed.status is OutcomeStatus.PASS
    assert passed.stage_outcomes == (syntax.ref, verified.ref)
    wrong_type = run_gate(pack, artifact, "accept_syntax", {"syntax": object()})  # type: ignore[dict-item]
    assert wrong_type.status is OutcomeStatus.BLOCK


def test_promotion_ledger_advances_only_on_exact_pass_chain() -> None:
    desc = descriptor()
    pack = ExamplePack(desc)
    artifact = candidate(desc)
    syntax = run_stage(pack, artifact, "syntax")
    first_gate = run_gate(pack, artifact, "accept_syntax", {"syntax": syntax})
    empty = PromotionLedger.create(artifact)
    first = empty.promote(desc, artifact, first_gate, {"syntax": syntax})
    verified = run_stage(pack, artifact, "verified", {"syntax": syntax})
    second_gate = run_gate(
        pack, artifact, "admit_verified", {"syntax": syntax, "verified": verified}
    )
    second = first.promote(desc, artifact, second_gate, {"syntax": syntax, "verified": verified})
    assert empty.current_stage is None
    assert first.current_stage == "syntax"
    assert second.current_stage == "verified"
    assert second.entries[1].prior_entry_sha256 == second.entries[0].entry_sha256
    assert PromotionLedger.from_dict(second.to_dict()) == second


@pytest.mark.parametrize("failure", ["block", "wrong_hash", "missing", "wrong_candidate"])
def test_promotion_fail_closed_preserves_original_ledger(failure: str) -> None:
    desc = descriptor()
    pack = ExamplePack(desc)
    artifact = candidate(desc)
    syntax = passed_stage(artifact, "syntax")
    gate = run_gate(pack, artifact, "accept_syntax", {"syntax": syntax})
    ledger = PromotionLedger.create(artifact)
    stages: dict[str, StageOutcome] = {"syntax": syntax}
    if failure == "block":
        gate = GateOutcome.create(
            "accept_syntax",
            artifact.ref,
            OutcomeStatus.BLOCK,
            (syntax.ref,),
            (CheckResult.create("blocked", False, {"ok": False}),),
            reason_codes=("missing_premise",),
        )
    elif failure == "wrong_hash":
        gate = GateOutcome.create(
            "accept_syntax",
            artifact.ref,
            OutcomeStatus.PASS,
            (OutcomeRef("syntax", ZERO),),
            (CheckResult.create("promotion_contract", True, {"ok": True}),),
        )
    elif failure == "missing":
        stages = {}
    else:
        other = candidate(desc, kind=ArtifactKind.THEOREM)
        gate = GateOutcome.create(
            "accept_syntax",
            other.ref,
            OutcomeStatus.PASS,
            (syntax.ref,),
            (CheckResult.create("promotion_contract", True, {"ok": True}),),
        )
    before = ledger.to_dict()
    with pytest.raises(PromotionDenied):
        ledger.promote(desc, artifact, gate, stages)
    assert ledger.to_dict() == before


@pytest.mark.parametrize("target", ["entry", "ledger", "artifact", "unknown"])
def test_resealed_ledger_tampering_still_fails_closed(target: str) -> None:
    desc = descriptor()
    pack = ExamplePack(desc)
    artifact = candidate(desc)
    syntax = passed_stage(artifact, "syntax")
    gate = run_gate(pack, artifact, "accept_syntax", {"syntax": syntax})
    value = (
        PromotionLedger.create(artifact).promote(desc, artifact, gate, {"syntax": syntax}).to_dict()
    )
    if target == "entry":
        value["entries"][0]["to_stage"] = "verified"
    elif target == "ledger":
        value["ledger_sha256"] = ZERO
    elif target == "artifact":
        value["artifact"]["content_sha256"] = ZERO
    else:
        value["promoted"] = True
    with pytest.raises(SchemaViolation):
        PromotionLedger.from_dict(value)


@pytest.mark.parametrize("target", ["status", "check", "stage_hash", "unknown"])
def test_resealed_outcome_semantic_tampering_fails_closed(target: str) -> None:
    artifact = candidate()
    syntax = passed_stage(artifact, "syntax")
    gate = GateOutcome.create(
        "accept_syntax",
        artifact.ref,
        OutcomeStatus.PASS,
        (syntax.ref,),
        (CheckResult.create("promotion_contract", True, {"ok": True}),),
    )
    value = gate.to_dict()
    if target == "status":
        value["status"] = "block"
    elif target == "check":
        value["checks"][0]["passed"] = False
    elif target == "stage_hash":
        value["stage_outcomes"][0]["outcome_sha256"] = ZERO
    else:
        value["origin"] = "trusted"
    if target != "unknown":
        value = reseal_gate(value)
    if target == "stage_hash":
        forged = GateOutcome.from_dict(value)
        with pytest.raises(PromotionDenied, match="stage outcome hashes"):
            PromotionLedger.create(artifact).promote(
                descriptor(), artifact, forged, {"syntax": syntax}
            )
        return
    with pytest.raises(SchemaViolation):
        GateOutcome.from_dict(value)


def test_stage_resealed_artifact_rebinding_fails_closed_at_promotion() -> None:
    desc = descriptor()
    artifact = candidate(desc)
    syntax = passed_stage(artifact, "syntax")
    value = syntax.to_dict()
    value["artifact"]["content_sha256"] = ZERO
    forged = StageOutcome.from_dict(reseal_stage(value))
    gate = GateOutcome.create(
        "accept_syntax",
        artifact.ref,
        OutcomeStatus.PASS,
        (forged.ref,),
        (CheckResult.create("promotion_contract", True, {"ok": True}),),
    )
    with pytest.raises(PromotionDenied, match="stage outcome"):
        PromotionLedger.create(artifact).promote(desc, artifact, gate, {"syntax": forged})
