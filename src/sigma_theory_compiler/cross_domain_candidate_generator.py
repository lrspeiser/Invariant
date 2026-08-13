"""Bounded Sigma-Core-native structural transfers across distinct domain packs.

The registry in this module packages already-existing candidate structures.  It does not
establish semantic truth, equivalence, novelty, fitness, or promotion eligibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    DomainPackRef,
    ProvenanceRecord,
    canonical_sha256,
)

RESULT_SCHEMA = "sigma-cross-domain-transfer-result-1.0"
TRANSFER_REPRESENTATION_SCHEMA = "sigma-cross-domain-structural-transfer-1.0"

_TEMPLATES: dict[str, dict[str, str]] = {
    "algorithm_record_bundle_v1": {
        "accepted_kind": ArtifactKind.ALGORITHM.value,
        "output_kind": ArtifactKind.CONSTRUCTION.value,
        "representation_schema": "algorithm-interface-steps-v1",
        "transfer_shape": "ordered_algorithm_records",
        "statement": "Package exact algorithm interface and step records by source domain.",
    },
    "formula_record_bundle_v1": {
        "accepted_kind": ArtifactKind.FORMULA.value,
        "output_kind": ArtifactKind.CONSTRUCTION.value,
        "representation_schema": "formula-expression-variables-v1",
        "transfer_shape": "ordered_formula_records",
        "statement": "Package exact formula expression and variable records by source domain.",
    },
    "formula_variable_index_v1": {
        "accepted_kind": ArtifactKind.FORMULA.value,
        "output_kind": ArtifactKind.CONSTRUCTION.value,
        "representation_schema": "formula-expression-variables-v1",
        "transfer_shape": "source_variable_index",
        "statement": "Package a source-domain index of exact formula variable records.",
    },
}
REGISTERED_TRANSFER_TEMPLATE_IDS = tuple(sorted(_TEMPLATES))

_NO_SEMANTIC_ASSUMPTIONS = (
    "Source candidates are bound by exact Sigma Core identities and are not revalidated semantically.",
    "This artifact is a structural container and makes no semantic truth or equivalence claim.",
)
_CLAIM_BOUNDARY = {
    "semantic_truth_claimed": False,
    "equivalence_claimed": False,
    "novelty_claimed": False,
    "promotion_authorized": False,
}


class CrossDomainTransferError(ValueError):
    """A request or sealed result crossed the structural-transfer boundary."""


@dataclass(frozen=True, slots=True)
class TransferLimits:
    maximum_parents: int = 16
    maximum_templates: int = 8
    maximum_candidates: int = 8
    maximum_work_units: int = 16

    def __post_init__(self) -> None:
        for name in (
            "maximum_parents",
            "maximum_templates",
            "maximum_candidates",
            "maximum_work_units",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CrossDomainTransferError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_parents": self.maximum_parents,
            "maximum_templates": self.maximum_templates,
            "maximum_candidates": self.maximum_candidates,
            "maximum_work_units": self.maximum_work_units,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TransferLimits:
        _exact_keys(
            value,
            {
                "maximum_parents",
                "maximum_templates",
                "maximum_candidates",
                "maximum_work_units",
            },
            "transfer limits",
        )
        return cls(**value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CrossDomainTransferError(f"{label} keys changed")


def _template_manifest() -> list[dict[str, str]]:
    rows = []
    for template_id in REGISTERED_TRANSFER_TEMPLATE_IDS:
        body = {"template_id": template_id, **_TEMPLATES[template_id]}
        rows.append({**body, "template_sha256": canonical_sha256(body)})
    return rows


def _template_sha256(template_id: str) -> str:
    body = {"template_id": template_id, **_TEMPLATES[template_id]}
    return canonical_sha256(body)


def _normalize_templates(
    template_ids: Sequence[str], limits: TransferLimits
) -> tuple[tuple[str, ...], int]:
    raw = tuple(template_ids)
    if not raw:
        raise CrossDomainTransferError("at least one transfer template must be selected")
    if len(raw) > limits.maximum_templates:
        raise CrossDomainTransferError("template inputs exceed maximum_templates")
    if any(not isinstance(item, str) for item in raw):
        raise CrossDomainTransferError("transfer template IDs must be strings")
    unknown = sorted(set(raw) - set(REGISTERED_TRANSFER_TEMPLATE_IDS))
    if unknown:
        raise CrossDomainTransferError(f"unknown structural transfer template: {unknown[0]}")
    return tuple(sorted(set(raw))), len(raw)


def _parent_sort_key(candidate: CandidateArtifact) -> tuple[str, str, str, str]:
    pack = candidate.provenance.domain_pack
    return pack.pack_id, pack.pack_version, pack.descriptor_sha256, candidate.content_sha256


def _normalize_parents(
    parents: Sequence[CandidateArtifact], target_pack: DomainPackRef, limits: TransferLimits
) -> tuple[tuple[CandidateArtifact, ...], int]:
    if not isinstance(target_pack, DomainPackRef):
        raise CrossDomainTransferError("target_pack must be a Sigma Core DomainPackRef")
    raw = tuple(parents)
    if len(raw) > limits.maximum_parents:
        raise CrossDomainTransferError("parent inputs exceed maximum_parents")
    if len(raw) < 2:
        raise CrossDomainTransferError("cross-domain transfer requires at least two parents")
    unique: dict[str, CandidateArtifact] = {}
    for parent in raw:
        if not isinstance(parent, CandidateArtifact):
            raise CrossDomainTransferError("every parent must be a Sigma Core CandidateArtifact")
        parent.validate()
        previous = unique.get(parent.content_sha256)
        if previous is not None and previous.to_dict() != parent.to_dict():
            raise CrossDomainTransferError("parent content digest collision changed canonical body")
        unique[parent.content_sha256] = parent
    ordered = tuple(sorted(unique.values(), key=_parent_sort_key))
    if len(ordered) < 2:
        raise CrossDomainTransferError("cross-domain transfer needs two unique parents")
    source_pack_ids = [parent.provenance.domain_pack.pack_id for parent in ordered]
    if len(set(source_pack_ids)) != len(source_pack_ids):
        raise CrossDomainTransferError("source parents must come from distinct domain pack IDs")
    if target_pack.pack_id in source_pack_ids:
        raise CrossDomainTransferError("target domain pack must be distinct from every source pack")
    return ordered, len(raw)


def _nonempty_strings(value: Any, label: str, *, sorted_unique: bool) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CrossDomainTransferError(f"{label} must be a nonempty string array")
    if any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in value):
        raise CrossDomainTransferError(f"{label} must contain nonempty stripped strings")
    if sorted_unique and value != sorted(set(value)):
        raise CrossDomainTransferError(f"{label} must be sorted and unique")
    return value


def _validate_parent_representation(parent: CandidateArtifact, template_id: str) -> None:
    template = _TEMPLATES[template_id]
    if parent.kind.value != template["accepted_kind"]:
        raise CrossDomainTransferError(
            f"template {template_id} is incompatible with parent kind {parent.kind.value}"
        )
    representation = parent.representation
    if template["representation_schema"] == "formula-expression-variables-v1":
        _exact_keys(representation, {"expression", "variables"}, "formula representation")
        expression = representation["expression"]
        if (
            not isinstance(expression, str)
            or not expression.strip()
            or expression != expression.strip()
        ):
            raise CrossDomainTransferError("formula expression must be a nonempty stripped string")
        _nonempty_strings(representation["variables"], "formula variables", sorted_unique=True)
        return
    if template["representation_schema"] == "algorithm-interface-steps-v1":
        _exact_keys(
            representation,
            {"inputs", "outputs", "steps"},
            "algorithm representation",
        )
        _nonempty_strings(representation["inputs"], "algorithm inputs", sorted_unique=True)
        _nonempty_strings(representation["outputs"], "algorithm outputs", sorted_unique=True)
        _nonempty_strings(representation["steps"], "algorithm steps", sorted_unique=False)
        return
    raise CrossDomainTransferError("registered template representation schema is not implemented")


def _parent_binding(parent: CandidateArtifact) -> dict[str, Any]:
    return {
        "artifact_ref": parent.ref.to_dict(),
        "domain_pack": parent.provenance.domain_pack.to_dict(),
        "kind": parent.kind.value,
    }


def _transfer_payload(template_id: str, parents: Sequence[CandidateArtifact]) -> dict[str, Any]:
    shape = _TEMPLATES[template_id]["transfer_shape"]
    if shape == "ordered_formula_records":
        return {
            "transfer_shape": shape,
            "records": [
                {
                    "artifact_ref": parent.ref.to_dict(),
                    "expression": parent.representation["expression"],
                    "variables": list(parent.representation["variables"]),
                }
                for parent in parents
            ],
        }
    if shape == "source_variable_index":
        return {
            "transfer_shape": shape,
            "records": [
                {
                    "artifact_ref": parent.ref.to_dict(),
                    "variables": list(parent.representation["variables"]),
                }
                for parent in parents
            ],
        }
    if shape == "ordered_algorithm_records":
        return {
            "transfer_shape": shape,
            "records": [
                {
                    "artifact_ref": parent.ref.to_dict(),
                    "inputs": list(parent.representation["inputs"]),
                    "outputs": list(parent.representation["outputs"]),
                    "steps": list(parent.representation["steps"]),
                }
                for parent in parents
            ],
        }
    raise CrossDomainTransferError("registered transfer shape is not implemented")


def _create_candidate(
    template_id: str,
    parents: Sequence[CandidateArtifact],
    target_pack: DomainPackRef,
    registry_sha256: str,
) -> CandidateArtifact:
    template_sha256 = _template_sha256(template_id)
    bindings = [_parent_binding(parent) for parent in parents]
    parameters = {
        "generator_schema_version": RESULT_SCHEMA,
        "registry_sha256": registry_sha256,
        "template_id": template_id,
        "template_sha256": template_sha256,
        "target_domain_pack": target_pack.to_dict(),
        "ordered_parent_refs": [binding["artifact_ref"] for binding in bindings],
    }
    provenance = ProvenanceRecord.create(
        target_pack,
        parameters,
        inputs=tuple(parent.ref for parent in parents),
    )
    representation = {
        "schema_version": TRANSFER_REPRESENTATION_SCHEMA,
        "template_id": template_id,
        "template_sha256": template_sha256,
        "target_domain_pack": target_pack.to_dict(),
        "ordered_parent_bindings": bindings,
        "transferred_structure": _transfer_payload(template_id, parents),
    }
    statement = (
        f"Registered structural transfer {template_id} across {len(parents)} distinct source "
        f"domain packs into {target_pack.pack_id}."
    )
    return CandidateArtifact.create(
        ArtifactKind(_TEMPLATES[template_id]["output_kind"]),
        statement,
        representation,
        provenance,
        assumptions=_NO_SEMANTIC_ASSUMPTIONS,
        claims=(),
    )


def _receipt(
    sequence: int,
    work_unit: int,
    template_id: str,
    candidate: CandidateArtifact,
    parents: Sequence[CandidateArtifact],
    target_pack: DomainPackRef,
) -> dict[str, Any]:
    body = {
        "schema_version": "sigma-cross-domain-transfer-receipt-1.0",
        "sequence": sequence,
        "work_unit": work_unit,
        "template_id": template_id,
        "template_sha256": _template_sha256(template_id),
        "ordered_parent_refs": [parent.ref.to_dict() for parent in parents],
        "source_domain_packs": [parent.provenance.domain_pack.to_dict() for parent in parents],
        "target_domain_pack": target_pack.to_dict(),
        "candidate_ref": candidate.ref.to_dict(),
        "provenance_parameters_sha256": candidate.provenance.parameters_sha256,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def generate_cross_domain_candidates(
    parents: Sequence[CandidateArtifact],
    target_pack: DomainPackRef,
    *,
    template_ids: Sequence[str],
    limits: TransferLimits | None = None,
) -> dict[str, Any]:
    """Apply selected registered structural templates under deterministic work caps."""

    limits = TransferLimits() if limits is None else limits
    if not isinstance(limits, TransferLimits):
        raise CrossDomainTransferError("limits must be TransferLimits")
    ordered_parents, parent_input_count = _normalize_parents(parents, target_pack, limits)
    selected_templates, template_input_count = _normalize_templates(template_ids, limits)
    for template_id in selected_templates:
        for parent in ordered_parents:
            _validate_parent_representation(parent, template_id)

    registry = _template_manifest()
    registry_sha256 = canonical_sha256(registry)
    generated: dict[str, tuple[str, CandidateArtifact, int]] = {}
    work_units = 0
    decision = "completed_registered_structural_transfers"
    for template_id in selected_templates:
        if work_units >= limits.maximum_work_units:
            decision = "bounded_work_unit_cap"
            break
        if len(generated) >= limits.maximum_candidates:
            decision = "bounded_candidate_cap"
            break
        work_units += 1
        candidate = _create_candidate(template_id, ordered_parents, target_pack, registry_sha256)
        generated.setdefault(candidate.content_sha256, (template_id, candidate, work_units))

    ordered_generated = [generated[key] for key in sorted(generated)]
    candidates = [candidate.to_dict() for _, candidate, _ in ordered_generated]
    receipts = [
        _receipt(sequence, work_unit, template_id, candidate, ordered_parents, target_pack)
        for sequence, (template_id, candidate, work_unit) in enumerate(ordered_generated)
    ]
    parent_manifest = [_parent_binding(parent) for parent in ordered_parents]
    result = {
        "schema_version": RESULT_SCHEMA,
        "template_registry": registry,
        "template_registry_sha256": registry_sha256,
        "target_domain_pack": target_pack.to_dict(),
        "limits": limits.to_dict(),
        "selected_template_ids": list(selected_templates),
        "parent_manifest": parent_manifest,
        "candidates": candidates,
        "transfer_receipts": receipts,
        "transfer_receipt_root_sha256": canonical_sha256(receipts),
        "counts": {
            "parent_inputs": parent_input_count,
            "unique_parents": len(ordered_parents),
            "duplicate_parents_removed": parent_input_count - len(ordered_parents),
            "template_inputs": template_input_count,
            "unique_templates": len(selected_templates),
            "duplicate_templates_removed": template_input_count - len(selected_templates),
            "work_units_consumed": work_units,
            "candidates_emitted": len(candidates),
            "receipts_emitted": len(receipts),
        },
        "decision": decision,
        "claims": dict(_CLAIM_BOUNDARY),
    }
    result["content_sha256"] = canonical_sha256(result)
    validate_transfer_result(result)
    return result


def _validate_parent_manifest(
    value: Any, target_pack: DomainPackRef, limits: TransferLimits
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) < 2 or len(value) > limits.maximum_parents:
        raise CrossDomainTransferError("parent manifest cardinality violates limits")
    rows: list[Mapping[str, Any]] = []
    sort_keys: list[tuple[str, str, str, str]] = []
    pack_ids: list[str] = []
    refs: list[tuple[str, str]] = []
    for row in value:
        _exact_keys(row, {"artifact_ref", "domain_pack", "kind"}, "parent manifest row")
        pack = DomainPackRef.from_dict(row["domain_pack"])
        try:
            kind = ArtifactKind(row["kind"])
        except (TypeError, ValueError) as error:
            raise CrossDomainTransferError("parent manifest kind is unregistered") from error
        ref = row["artifact_ref"]
        _exact_keys(ref, {"artifact_id", "content_sha256"}, "parent artifact reference")
        artifact_id, digest = ref["artifact_id"], ref["content_sha256"]
        if not isinstance(artifact_id, str) or not isinstance(digest, str):
            raise CrossDomainTransferError("parent artifact reference types changed")
        if (
            artifact_id != f"sig-{digest[:24]}"
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CrossDomainTransferError("parent artifact reference identity changed")
        rows.append(row)
        pack_ids.append(pack.pack_id)
        refs.append((artifact_id, digest))
        sort_keys.append((pack.pack_id, pack.pack_version, pack.descriptor_sha256, digest))
        if kind.value != row["kind"]:
            raise CrossDomainTransferError("parent artifact kind changed")
    if sort_keys != sorted(sort_keys) or len(set(refs)) != len(refs):
        raise CrossDomainTransferError("parent manifest order or uniqueness changed")
    if len(set(pack_ids)) != len(pack_ids) or target_pack.pack_id in pack_ids:
        raise CrossDomainTransferError("parent/target domain separation changed")
    return rows


def _validate_transferred_structure(
    template_id: str, payload: Any, expected_refs: list[Mapping[str, Any]]
) -> None:
    _exact_keys(payload, {"transfer_shape", "records"}, "transferred structure")
    shape = _TEMPLATES[template_id]["transfer_shape"]
    if payload["transfer_shape"] != shape or not isinstance(payload["records"], list):
        raise CrossDomainTransferError("transferred structure shape changed")
    records = payload["records"]
    if len(records) != len(expected_refs):
        raise CrossDomainTransferError("transferred structure parent coverage changed")
    for record, expected_ref in zip(records, expected_refs, strict=True):
        if shape == "ordered_formula_records":
            _exact_keys(record, {"artifact_ref", "expression", "variables"}, "formula record")
            expression = record["expression"]
            if (
                not isinstance(expression, str)
                or not expression.strip()
                or expression != expression.strip()
            ):
                raise CrossDomainTransferError("formula record expression changed")
            _nonempty_strings(record["variables"], "formula record variables", sorted_unique=True)
        elif shape == "source_variable_index":
            _exact_keys(record, {"artifact_ref", "variables"}, "variable index record")
            _nonempty_strings(record["variables"], "variable index variables", sorted_unique=True)
        elif shape == "ordered_algorithm_records":
            _exact_keys(
                record,
                {"artifact_ref", "inputs", "outputs", "steps"},
                "algorithm record",
            )
            _nonempty_strings(record["inputs"], "algorithm record inputs", sorted_unique=True)
            _nonempty_strings(record["outputs"], "algorithm record outputs", sorted_unique=True)
            _nonempty_strings(record["steps"], "algorithm record steps", sorted_unique=False)
        else:
            raise CrossDomainTransferError("registered transfer shape is not implemented")
        if record["artifact_ref"] != expected_ref:
            raise CrossDomainTransferError("transferred structure parent reference changed")


def validate_transfer_result(value: Mapping[str, Any]) -> None:
    """Validate canonical bindings and receipt relationships without semantic claims."""

    expected_keys = {
        "schema_version",
        "template_registry",
        "template_registry_sha256",
        "target_domain_pack",
        "limits",
        "selected_template_ids",
        "parent_manifest",
        "candidates",
        "transfer_receipts",
        "transfer_receipt_root_sha256",
        "counts",
        "decision",
        "claims",
        "content_sha256",
    }
    _exact_keys(value, expected_keys, "transfer result")
    unsigned = {key: child for key, child in value.items() if key != "content_sha256"}
    if value["schema_version"] != RESULT_SCHEMA or value["content_sha256"] != canonical_sha256(
        unsigned
    ):
        raise CrossDomainTransferError("transfer result identity changed")
    registry = _template_manifest()
    if value["template_registry"] != registry or value[
        "template_registry_sha256"
    ] != canonical_sha256(registry):
        raise CrossDomainTransferError("registered transfer template manifest changed")
    target_pack = DomainPackRef.from_dict(value["target_domain_pack"])
    limits = TransferLimits.from_dict(value["limits"])
    selected = value["selected_template_ids"]
    if (
        not isinstance(selected, list)
        or not selected
        or selected != sorted(set(selected))
        or any(item not in REGISTERED_TRANSFER_TEMPLATE_IDS for item in selected)
        or len(selected) > limits.maximum_templates
    ):
        raise CrossDomainTransferError("selected transfer templates changed")
    parents = _validate_parent_manifest(value["parent_manifest"], target_pack, limits)
    expected_refs = [row["artifact_ref"] for row in parents]
    expected_inputs = sorted(expected_refs, key=lambda item: item["artifact_id"])

    candidates_raw = value["candidates"]
    receipts = value["transfer_receipts"]
    if not isinstance(candidates_raw, list) or not isinstance(receipts, list):
        raise CrossDomainTransferError("candidates and receipts must be arrays")
    candidates = [CandidateArtifact.from_dict(item) for item in candidates_raw]
    if [item.content_sha256 for item in candidates] != sorted(
        {item.content_sha256 for item in candidates}
    ):
        raise CrossDomainTransferError("candidate ordering or canonical dedup changed")
    if len(candidates) > limits.maximum_candidates or len(candidates) != len(receipts):
        raise CrossDomainTransferError("candidate or receipt cardinality changed")

    receipt_templates: list[str] = []
    for sequence, (candidate, receipt) in enumerate(zip(candidates, receipts, strict=True)):
        representation = candidate.representation
        _exact_keys(
            representation,
            {
                "schema_version",
                "template_id",
                "template_sha256",
                "target_domain_pack",
                "ordered_parent_bindings",
                "transferred_structure",
            },
            "transferred candidate representation",
        )
        template_id = representation["template_id"]
        if template_id not in selected:
            raise CrossDomainTransferError("candidate used an unselected transfer template")
        expected_statement = (
            f"Registered structural transfer {template_id} across {len(parents)} distinct source "
            f"domain packs into {target_pack.pack_id}."
        )
        expected_parameters = {
            "generator_schema_version": RESULT_SCHEMA,
            "registry_sha256": canonical_sha256(registry),
            "template_id": template_id,
            "template_sha256": _template_sha256(template_id),
            "target_domain_pack": target_pack.to_dict(),
            "ordered_parent_refs": expected_refs,
        }
        if (
            representation["schema_version"] != TRANSFER_REPRESENTATION_SCHEMA
            or representation["template_sha256"] != _template_sha256(template_id)
            or representation["target_domain_pack"] != target_pack.to_dict()
            or representation["ordered_parent_bindings"] != parents
        ):
            raise CrossDomainTransferError("transferred candidate bindings changed")
        if (
            candidate.kind.value != _TEMPLATES[template_id]["output_kind"]
            or candidate.statement != expected_statement
            or candidate.claims
            or candidate.assumptions != tuple(sorted(_NO_SEMANTIC_ASSUMPTIONS))
            or candidate.provenance.domain_pack != target_pack
            or [item.to_dict() for item in candidate.provenance.inputs] != expected_inputs
            or candidate.provenance.sources
            or candidate.provenance.parameters_sha256 != canonical_sha256(expected_parameters)
        ):
            raise CrossDomainTransferError("candidate provenance or claim boundary changed")
        if any(row["kind"] != _TEMPLATES[template_id]["accepted_kind"] for row in parents):
            raise CrossDomainTransferError("candidate source kind compatibility changed")
        _validate_transferred_structure(
            template_id, representation["transferred_structure"], expected_refs
        )

        expected_receipt_keys = {
            "schema_version",
            "sequence",
            "work_unit",
            "template_id",
            "template_sha256",
            "ordered_parent_refs",
            "source_domain_packs",
            "target_domain_pack",
            "candidate_ref",
            "provenance_parameters_sha256",
            "receipt_sha256",
        }
        _exact_keys(receipt, expected_receipt_keys, "transfer receipt")
        receipt_body = {key: child for key, child in receipt.items() if key != "receipt_sha256"}
        if (
            receipt["receipt_sha256"] != canonical_sha256(receipt_body)
            or receipt["schema_version"] != "sigma-cross-domain-transfer-receipt-1.0"
            or receipt["sequence"] != sequence
            or receipt["work_unit"] != selected.index(template_id) + 1
            or receipt["template_id"] != template_id
            or receipt["template_sha256"] != _template_sha256(template_id)
            or receipt["ordered_parent_refs"] != expected_refs
            or receipt["source_domain_packs"] != [row["domain_pack"] for row in parents]
            or receipt["target_domain_pack"] != target_pack.to_dict()
            or receipt["candidate_ref"] != candidate.ref.to_dict()
            or receipt["provenance_parameters_sha256"] != candidate.provenance.parameters_sha256
        ):
            raise CrossDomainTransferError("transfer receipt binding changed")
        receipt_templates.append(template_id)
    if len(set(receipt_templates)) != len(receipt_templates):
        raise CrossDomainTransferError("transfer templates emitted duplicate candidates")
    if set(receipt_templates) != set(selected[: len(candidates)]):
        raise CrossDomainTransferError("emitted transfer template prefix changed")
    if value["transfer_receipt_root_sha256"] != canonical_sha256(receipts):
        raise CrossDomainTransferError("transfer receipt root changed")

    counts = value["counts"]
    _exact_keys(
        counts,
        {
            "parent_inputs",
            "unique_parents",
            "duplicate_parents_removed",
            "template_inputs",
            "unique_templates",
            "duplicate_templates_removed",
            "work_units_consumed",
            "candidates_emitted",
            "receipts_emitted",
        },
        "transfer counts",
    )
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in counts.values()
    ):
        raise CrossDomainTransferError("transfer counts must be nonnegative integers")
    if (
        counts["unique_parents"] != len(parents)
        or counts["parent_inputs"] > limits.maximum_parents
        or counts["parent_inputs"] - counts["duplicate_parents_removed"] != len(parents)
        or counts["unique_templates"] != len(selected)
        or counts["template_inputs"] > limits.maximum_templates
        or counts["template_inputs"] - counts["duplicate_templates_removed"] != len(selected)
        or counts["work_units_consumed"] != len(candidates)
        or counts["candidates_emitted"] != len(candidates)
        or counts["receipts_emitted"] != len(receipts)
        or counts["work_units_consumed"] > limits.maximum_work_units
    ):
        raise CrossDomainTransferError("transfer counts do not reconcile")
    remaining = len(selected) - counts["work_units_consumed"]
    if remaining == 0:
        expected_decision = "completed_registered_structural_transfers"
    elif counts["work_units_consumed"] >= limits.maximum_work_units:
        expected_decision = "bounded_work_unit_cap"
    elif len(candidates) >= limits.maximum_candidates:
        expected_decision = "bounded_candidate_cap"
    else:
        raise CrossDomainTransferError("incomplete transfer has no exhausted bound")
    if value["decision"] != expected_decision or value["claims"] != _CLAIM_BOUNDARY:
        raise CrossDomainTransferError("transfer decision or claim boundary changed")


def validate_transfer_replay(
    value: Mapping[str, Any],
    parents: Sequence[CandidateArtifact],
    target_pack: DomainPackRef,
    *,
    template_ids: Sequence[str],
    limits: TransferLimits | None = None,
) -> None:
    """Re-run the bounded transfer and require byte-semantically identical canonical data."""

    validate_transfer_result(value)
    replayed = generate_cross_domain_candidates(
        parents,
        target_pack,
        template_ids=template_ids,
        limits=limits,
    )
    if replayed != value:
        raise CrossDomainTransferError("transfer result is not replayable from supplied parents")
