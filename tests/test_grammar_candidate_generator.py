from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import sigma_theory_compiler.grammar_candidate_generator as adapter
from sigma_theory_compiler.grammar import enumerate_expressions
from sigma_theory_compiler.grammar_candidate_generator import (
    GrammarGenerationManifest,
    GrammarLimits,
    GrammarSpec,
    generate_grammar_candidates,
    grammar_source_bindings,
    validate_grammar_manifest,
)
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    DomainPackRef,
    OutcomeStatus,
    SchemaViolation,
    SourceBinding,
    canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = DomainPackRef("synthetic.grammar", "1.0", "1" * 64)
SPEC = GrammarSpec(("q", "x"), ("saturate",), ("add", "multiply"), 3)
LIMITS = GrammarLimits(100, 1_000, 1_000)


def _sources() -> tuple[SourceBinding, ...]:
    return grammar_source_bindings(ROOT)


def _manifest(
    spec: GrammarSpec = SPEC, limits: GrammarLimits = LIMITS
) -> GrammarGenerationManifest:
    return generate_grammar_candidates(spec, limits, DOMAIN, _sources())


def _reseal_candidate(candidate: dict[str, object]) -> None:
    body = {
        key: value
        for key, value in candidate.items()
        if key not in {"artifact_id", "content_sha256"}
    }
    digest = canonical_sha256(body)
    candidate["artifact_id"] = f"sig-{digest[:24]}"
    candidate["content_sha256"] = digest


def _reseal_lineage(record: dict[str, object]) -> None:
    body = {key: value for key, value in record.items() if key != "lineage_sha256"}
    record["lineage_sha256"] = canonical_sha256(body)


def _reseal_manifest(value: dict[str, object]) -> None:
    body = {key: child for key, child in value.items() if key != "manifest_sha256"}
    value["manifest_sha256"] = canonical_sha256(body)


def test_matches_legacy_bounded_semantics_and_order() -> None:
    manifest = _manifest()
    legacy, statistics = enumerate_expressions(
        list(SPEC.atoms),
        list(SPEC.unary_operators),
        list(SPEC.binary_operators),
        SPEC.maximum_complexity,
    )

    assert manifest.status is OutcomeStatus.PASS
    assert [item.representation["canonical_sympy"] for item in manifest.candidates] == [
        item.canonical for item in legacy
    ]
    assert [item.representation["complexity"] for item in manifest.candidates] == [
        item.complexity for item in legacy
    ]
    assert manifest.counts.work_units == statistics["generated_before_deduplication"] == 14
    assert manifest.counts.unique_discovered == statistics["unique"] == 12
    assert manifest.counts.duplicates_observed == statistics["duplicates_removed"] == 2
    assert manifest.counts.emitted_nodes == 30


def test_success_emits_sigma_core_formula_artifacts_with_exact_provenance() -> None:
    manifest = _manifest()
    spec_hash = canonical_sha256(SPEC.to_dict())

    assert manifest.candidates
    assert all(item.kind is ArtifactKind.FORMULA for item in manifest.candidates)
    assert all(item.claims == ("bounded_grammar_candidate",) for item in manifest.candidates)
    assert all(
        item.representation["grammar_spec_sha256"] == spec_hash for item in manifest.candidates
    )
    assert all(item.provenance.domain_pack == DOMAIN for item in manifest.candidates)
    assert all(item.provenance.sources == _sources() for item in manifest.candidates)
    assert len({item.provenance.parameters_sha256 for item in manifest.candidates}) == len(
        manifest.candidates
    )
    by_child = {item.child: item for item in manifest.lineage}
    for candidate in manifest.candidates:
        expected_inputs = tuple(
            sorted(set(by_child[candidate.ref].parents), key=lambda item: item.artifact_id)
        )
        assert candidate.provenance.inputs == expected_inputs


def test_manifest_scientific_boundary_is_explicitly_closed() -> None:
    value = _manifest().to_dict()

    assert value["generation_only"] is True
    assert value["truth_established"] is False
    assert value["promotion_allowed"] is False
    assert all("truth" not in claim for item in value["candidates"] for claim in item["claims"])


def test_lineage_is_contiguous_closed_and_replayable() -> None:
    manifest = _manifest()
    known = {item.artifact_id for item in manifest.candidates}

    assert [item.ordinal for item in manifest.lineage] == list(range(12))
    assert [item.operation for item in manifest.lineage[:2]] == ["atom", "atom"]
    for item in manifest.lineage:
        assert item.child.artifact_id in known
        assert all(parent.artifact_id in known for parent in item.parents)
        assert len(item.parents) == {"atom": 0, "unary": 1, "binary": 2}[item.operation]
    assert validate_grammar_manifest(manifest, project_root=ROOT) == manifest


def test_generation_and_serialization_are_deterministic() -> None:
    first = _manifest()
    second = _manifest()

    assert first.to_dict() == second.to_dict()
    assert GrammarGenerationManifest.from_dict(first.to_dict()) == first
    assert first.manifest_sha256 == second.manifest_sha256
    assert len({item.content_sha256 for item in first.candidates}) == len(first.candidates)


@pytest.mark.parametrize(
    ("spec", "reasons"),
    [
        (GrammarSpec(("q", "unknown"), (), (), 1), ("unknown_atom",)),
        (GrammarSpec(("q",), ("mystery",), (), 2), ("unknown_unary_operator",)),
        (GrammarSpec(("q",), (), ("divide",), 3), ("unknown_binary_operator",)),
        (
            GrammarSpec(("unknown",), ("mystery",), ("divide",), 3),
            ("unknown_atom", "unknown_binary_operator", "unknown_unary_operator"),
        ),
    ],
)
def test_unknown_grammar_names_fail_closed_with_typed_reject(
    spec: GrammarSpec, reasons: tuple[str, ...]
) -> None:
    manifest = _manifest(spec)

    assert manifest.status is OutcomeStatus.REJECT
    assert manifest.reason_codes == reasons
    assert manifest.candidates == manifest.lineage == ()
    assert manifest.counts.to_dict() == dict.fromkeys(manifest.counts.to_dict(), 0)
    assert validate_grammar_manifest(manifest) == manifest


@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        (GrammarLimits(1, 1_000, 1_000), "expression_cap_reached"),
        (GrammarLimits(100, 1, 1_000), "node_cap_reached"),
        (GrammarLimits(100, 1_000, 1), "work_cap_reached"),
    ],
)
def test_caps_block_without_partial_artifacts(limits: GrammarLimits, reason: str) -> None:
    manifest = _manifest(limits=limits)

    assert manifest.status is OutcomeStatus.BLOCK
    assert manifest.reason_codes == (reason,)
    assert manifest.candidates == manifest.lineage == ()
    assert manifest.counts.emitted_expressions == manifest.counts.emitted_nodes == 0
    assert validate_grammar_manifest(manifest) == manifest


@pytest.mark.parametrize("value", [0, -1, True, 250_001])
def test_limits_reject_nonpositive_boolean_and_above_hard_ceiling(value: int) -> None:
    with pytest.raises(SchemaViolation):
        GrammarLimits(value, 10, 10)


def test_spec_rejects_order_drift_duplicates_and_empty_atoms() -> None:
    with pytest.raises(SchemaViolation):
        GrammarSpec(("x", "q"), (), (), 1)
    with pytest.raises(SchemaViolation):
        GrammarSpec(("q", "q"), (), (), 1)
    with pytest.raises(SchemaViolation):
        GrammarSpec((), (), (), 1)


def test_source_bindings_hash_exact_live_files() -> None:
    bindings = _sources()

    assert tuple(item.role for item in bindings) == ("adapter", "legacy_grammar")
    for item in bindings:
        assert item.file_sha256 == hashlib.sha256((ROOT / item.path).read_bytes()).hexdigest()


def test_wrong_source_roles_paths_and_live_hash_fail_closed() -> None:
    sources = list(_sources())
    with pytest.raises(SchemaViolation):
        generate_grammar_candidates(SPEC, LIMITS, DOMAIN, sources[:1])
    sources[0] = SourceBinding(
        sources[0].role, "src/sigma_theory_compiler/grammar.py", sources[0].file_sha256
    )
    with pytest.raises(SchemaViolation):
        generate_grammar_candidates(SPEC, LIMITS, DOMAIN, sources)

    value = _manifest().to_dict()
    value["sources"][0]["file_sha256"] = "f" * 64
    _reseal_manifest(value)
    with pytest.raises(SchemaViolation, match="source bytes changed"):
        validate_grammar_manifest(value, project_root=ROOT)


def test_unknown_manifest_key_and_scientific_claim_tamper_fail_closed() -> None:
    value = _manifest().to_dict()
    value["extra"] = "unregistered"
    with pytest.raises(SchemaViolation, match="keys changed"):
        validate_grammar_manifest(value)

    for key, replacement in (
        ("generation_only", False),
        ("truth_established", True),
        ("promotion_allowed", True),
    ):
        value = _manifest().to_dict()
        value[key] = replacement
        _reseal_manifest(value)
        with pytest.raises(SchemaViolation, match="scientific boundary"):
            validate_grammar_manifest(value)


def test_resealed_candidate_tamper_is_caught_by_full_replay() -> None:
    value = _manifest().to_dict()
    candidate = value["candidates"][0]
    candidate["statement"] = "A different but internally resealed statement."
    old_id = candidate["artifact_id"]
    _reseal_candidate(candidate)
    for record in value["lineage"]:
        if record["child"]["artifact_id"] == old_id:
            record["child"] = {
                "artifact_id": candidate["artifact_id"],
                "content_sha256": candidate["content_sha256"],
            }
            _reseal_lineage(record)
    _reseal_manifest(value)

    with pytest.raises(SchemaViolation):
        validate_grammar_manifest(value)


def test_resealed_counts_and_lineage_tampers_are_caught() -> None:
    counts = _manifest().to_dict()
    counts["counts"]["work_units"] += 1
    counts["counts"]["duplicates_observed"] += 1
    _reseal_manifest(counts)
    with pytest.raises(SchemaViolation, match="deterministic replay"):
        validate_grammar_manifest(counts)

    lineage = _manifest().to_dict()
    record = lineage["lineage"][2]
    record["operator"] = "sqrt1p_minus1"
    _reseal_lineage(record)
    _reseal_manifest(lineage)
    with pytest.raises(SchemaViolation, match="deterministic replay"):
        validate_grammar_manifest(lineage)


def test_resealed_spec_and_domain_tampers_are_caught_by_replay_or_identity() -> None:
    spec = _manifest().to_dict()
    spec["spec"]["maximum_complexity"] = 2
    _reseal_manifest(spec)
    with pytest.raises(SchemaViolation):
        validate_grammar_manifest(spec)

    domain = _manifest().to_dict()
    domain["domain_pack"]["descriptor_sha256"] = "2" * 64
    _reseal_manifest(domain)
    with pytest.raises(SchemaViolation):
        validate_grammar_manifest(domain)


def test_symbolic_backend_failure_becomes_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_canonicalize(_expression: object) -> tuple[object, str]:
        raise RuntimeError("backend detail must not enter the manifest")

    monkeypatch.setattr(adapter, "canonicalize", broken_canonicalize)
    manifest = _manifest()

    assert manifest.status is OutcomeStatus.ERROR
    assert manifest.reason_codes == ("canonicalization_error",)
    assert manifest.candidates == manifest.lineage == ()


def test_from_dict_detaches_mutable_payloads() -> None:
    value = _manifest().to_dict()
    parsed = GrammarGenerationManifest.from_dict(copy.deepcopy(value))
    value["candidates"][0]["representation"]["expression"] = "tampered"

    assert parsed.to_dict() != value
