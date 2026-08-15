from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.cross_domain_candidate_generator import (
    REGISTERED_TRANSFER_TEMPLATE_IDS,
    CrossDomainTransferError,
    TransferLimits,
    generate_cross_domain_candidates,
    validate_transfer_replay,
    validate_transfer_result,
)
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    DomainPackRef,
    ProvenanceRecord,
    canonical_sha256,
)


def _pack(pack_id: str) -> DomainPackRef:
    return DomainPackRef(pack_id, "1.0", canonical_sha256({"pack_id": pack_id, "version": "1.0"}))


def _candidate(
    pack_id: str,
    kind: ArtifactKind,
    representation: dict[str, object],
    statement: str,
) -> CandidateArtifact:
    pack = _pack(pack_id)
    provenance = ProvenanceRecord.create(pack, {"fixture": statement})
    return CandidateArtifact.create(kind, statement, representation, provenance)


@pytest.fixture
def formula_parents() -> tuple[CandidateArtifact, CandidateArtifact]:
    return (
        _candidate(
            "domain.geometry",
            ArtifactKind.FORMULA,
            {"expression": "A(r)=4*pi*r^2", "variables": ["r"]},
            "Surface area record.",
        ),
        _candidate(
            "domain.statistics",
            ArtifactKind.FORMULA,
            {"expression": "V(X)=E[(X-mu)^2]", "variables": ["X", "mu"]},
            "Variance record.",
        ),
    )


def test_registered_cross_domain_transfer_is_deterministic_and_replayable(
    formula_parents: tuple[CandidateArtifact, CandidateArtifact],
) -> None:
    target = _pack("domain.structural-synthesis")
    templates = ("formula_record_bundle_v1", "formula_variable_index_v1")
    limits = TransferLimits(8, 4, 4, 4)
    result = generate_cross_domain_candidates(
        formula_parents,
        target,
        template_ids=templates,
        limits=limits,
    )
    assert result == generate_cross_domain_candidates(
        tuple(reversed(formula_parents)),
        target,
        template_ids=tuple(reversed(templates)),
        limits=limits,
    )
    validate_transfer_result(result)
    validate_transfer_replay(result, formula_parents, target, template_ids=templates, limits=limits)
    assert result["decision"] == "completed_registered_structural_transfers"
    assert result["counts"] == {
        "parent_inputs": 2,
        "unique_parents": 2,
        "duplicate_parents_removed": 0,
        "template_inputs": 2,
        "unique_templates": 2,
        "duplicate_templates_removed": 0,
        "work_units_consumed": 2,
        "candidates_emitted": 2,
        "receipts_emitted": 2,
    }
    assert result["claims"] == {
        "semantic_truth_claimed": False,
        "equivalence_claimed": False,
        "novelty_claimed": False,
        "promotion_authorized": False,
    }


def test_outputs_are_sigma_core_native_with_exact_parent_and_target_provenance(
    formula_parents: tuple[CandidateArtifact, CandidateArtifact],
) -> None:
    target = _pack("domain.structural-synthesis")
    result = generate_cross_domain_candidates(
        formula_parents,
        target,
        template_ids=("formula_record_bundle_v1",),
    )
    candidate = CandidateArtifact.from_dict(result["candidates"][0])
    candidate.validate()
    assert candidate.kind is ArtifactKind.CONSTRUCTION
    assert candidate.provenance.domain_pack == target
    assert [item.to_dict() for item in candidate.provenance.inputs] == sorted(
        [parent.ref.to_dict() for parent in formula_parents], key=lambda item: item["artifact_id"]
    )
    assert candidate.claims == ()
    assert "no semantic truth or equivalence claim" in " ".join(candidate.assumptions)
    receipt = result["transfer_receipts"][0]
    assert receipt["candidate_ref"] == candidate.ref.to_dict()
    assert result["transfer_receipt_root_sha256"] == canonical_sha256(result["transfer_receipts"])


def test_parent_and_template_inputs_are_canonically_deduplicated(
    formula_parents: tuple[CandidateArtifact, CandidateArtifact],
) -> None:
    result = generate_cross_domain_candidates(
        (*formula_parents, formula_parents[0]),
        _pack("domain.structural-synthesis"),
        template_ids=("formula_record_bundle_v1", "formula_record_bundle_v1"),
    )
    assert result["counts"]["duplicate_parents_removed"] == 1
    assert result["counts"]["duplicate_templates_removed"] == 1
    assert result["counts"]["candidates_emitted"] == 1


def test_same_domain_and_target_domain_misuse_fail_closed(
    formula_parents: tuple[CandidateArtifact, CandidateArtifact],
) -> None:
    same_domain = _candidate(
        "domain.geometry",
        ArtifactKind.FORMULA,
        {"expression": "C(r)=2*pi*r", "variables": ["r"]},
        "Circumference record.",
    )
    with pytest.raises(CrossDomainTransferError, match="distinct domain pack IDs"):
        generate_cross_domain_candidates(
            (formula_parents[0], same_domain),
            _pack("domain.structural-synthesis"),
            template_ids=("formula_record_bundle_v1",),
        )
    with pytest.raises(CrossDomainTransferError, match="target domain pack must be distinct"):
        generate_cross_domain_candidates(
            formula_parents,
            _pack("domain.geometry"),
            template_ids=("formula_record_bundle_v1",),
        )


def test_unknown_templates_and_incompatible_kinds_or_representations_are_rejected(
    formula_parents: tuple[CandidateArtifact, CandidateArtifact],
) -> None:
    target = _pack("domain.structural-synthesis")
    with pytest.raises(CrossDomainTransferError, match="unknown structural transfer template"):
        generate_cross_domain_candidates(
            formula_parents, target, template_ids=("semantic_analogy_v1",)
        )
    with pytest.raises(CrossDomainTransferError, match="incompatible with parent kind"):
        generate_cross_domain_candidates(
            formula_parents, target, template_ids=("algorithm_record_bundle_v1",)
        )
    malformed = _candidate(
        "domain.algebra",
        ArtifactKind.FORMULA,
        {"expression": "x+1", "free_symbols": ["x"]},
        "Malformed transfer representation.",
    )
    with pytest.raises(CrossDomainTransferError, match="formula representation keys changed"):
        generate_cross_domain_candidates(
            (formula_parents[0], malformed),
            target,
            template_ids=("formula_record_bundle_v1",),
        )


def test_parent_template_candidate_and_work_bounds_are_fail_closed(
    formula_parents: tuple[CandidateArtifact, CandidateArtifact],
) -> None:
    target = _pack("domain.structural-synthesis")
    templates = ("formula_record_bundle_v1", "formula_variable_index_v1")
    with pytest.raises(CrossDomainTransferError, match="parent inputs exceed"):
        generate_cross_domain_candidates(
            (*formula_parents, formula_parents[0]),
            target,
            template_ids=(templates[0],),
            limits=TransferLimits(2, 4, 4, 4),
        )
    with pytest.raises(CrossDomainTransferError, match="template inputs exceed"):
        generate_cross_domain_candidates(
            formula_parents,
            target,
            template_ids=templates,
            limits=TransferLimits(4, 1, 4, 4),
        )
    candidate_limited = generate_cross_domain_candidates(
        formula_parents,
        target,
        template_ids=templates,
        limits=TransferLimits(4, 4, 1, 4),
    )
    assert candidate_limited["decision"] == "bounded_candidate_cap"
    assert candidate_limited["counts"]["candidates_emitted"] == 1
    work_limited = generate_cross_domain_candidates(
        formula_parents,
        target,
        template_ids=templates,
        limits=TransferLimits(4, 4, 4, 1),
    )
    assert work_limited["decision"] == "bounded_work_unit_cap"
    assert work_limited["counts"]["work_units_consumed"] == 1


def test_algorithm_template_preserves_ordered_interface_records() -> None:
    parents = (
        _candidate(
            "domain.compiler",
            ArtifactKind.ALGORITHM,
            {"inputs": ["ast"], "outputs": ["ir"], "steps": ["parse", "lower"]},
            "Compiler pipeline record.",
        ),
        _candidate(
            "domain.optimization",
            ArtifactKind.ALGORITHM,
            {"inputs": ["graph"], "outputs": ["plan"], "steps": ["score", "select"]},
            "Optimization pipeline record.",
        ),
    )
    result = generate_cross_domain_candidates(
        parents,
        _pack("domain.structural-synthesis"),
        template_ids=("algorithm_record_bundle_v1",),
    )
    candidate = CandidateArtifact.from_dict(result["candidates"][0])
    records = candidate.representation["transferred_structure"]["records"]
    assert len(records) == 2
    assert {tuple(record["steps"]) for record in records} == {
        ("parse", "lower"),
        ("score", "select"),
    }


def test_resealed_candidate_receipt_and_replay_tampering_are_rejected(
    formula_parents: tuple[CandidateArtifact, CandidateArtifact],
) -> None:
    target = _pack("domain.structural-synthesis")
    templates = ("formula_record_bundle_v1", "formula_variable_index_v1")
    result = generate_cross_domain_candidates(formula_parents, target, template_ids=templates)
    tampered = copy.deepcopy(result)
    tampered["transfer_receipts"][0]["template_id"] = "formula_variable_index_v1"
    receipt = tampered["transfer_receipts"][0]
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    tampered["transfer_receipt_root_sha256"] = canonical_sha256(tampered["transfer_receipts"])
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(CrossDomainTransferError, match="receipt binding changed"):
        validate_transfer_result(tampered)

    tampered = copy.deepcopy(result)
    candidate = tampered["candidates"][0]
    candidate["statement"] = "Unregistered semantic transfer claim."
    candidate_body = {
        key: candidate[key]
        for key in (
            "schema_version",
            "kind",
            "statement",
            "representation",
            "assumptions",
            "claims",
            "provenance",
        )
    }
    candidate["content_sha256"] = canonical_sha256(candidate_body)
    candidate["artifact_id"] = f"sig-{candidate['content_sha256'][:24]}"
    receipt = tampered["transfer_receipts"][0]
    receipt["candidate_ref"] = {
        "artifact_id": candidate["artifact_id"],
        "content_sha256": candidate["content_sha256"],
    }
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    tampered["transfer_receipt_root_sha256"] = canonical_sha256(tampered["transfer_receipts"])
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(CrossDomainTransferError, match="provenance or claim boundary"):
        validate_transfer_result(tampered)

    changed_parent = _candidate(
        "domain.statistics",
        ArtifactKind.FORMULA,
        {"expression": "SD(X)=sqrt(V(X))", "variables": ["X"]},
        "Changed statistics parent.",
    )
    with pytest.raises(CrossDomainTransferError, match="not replayable"):
        validate_transfer_replay(
            result,
            (formula_parents[0], changed_parent),
            target,
            template_ids=templates,
        )


def test_registered_template_registry_is_closed_and_sorted() -> None:
    assert REGISTERED_TRANSFER_TEMPLATE_IDS == (
        "algorithm_record_bundle_v1",
        "formula_record_bundle_v1",
        "formula_variable_index_v1",
    )
    with pytest.raises(CrossDomainTransferError, match="positive integer"):
        TransferLimits(maximum_work_units=0)
