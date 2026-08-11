from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g4_galaxy_manifest_bundle_tooling import (
    RECEIPT_SCHEMA,
    TOOLING_CONTRACT_SHA256,
    _sha,
    _synthetic_manifest,
    audit_dataset_manifest,
    build_g4_galaxy_manifest_bundle_tooling_readiness,
    build_prediction_bundle_draft,
    validate_prediction_bundle_draft,
    validate_tooling_contract,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY
from sigma_theory_compiler.reviewed_g4_candidate_galaxy_evaluator import (
    REQUIRED_REGISTRATION_HASHES,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g4_galaxy_manifest_bundle_tooling.json"
CONTRACT = ROOT / "configs" / "g4_galaxy_manifest_bundle_tooling_contract.json"
PREDECESSOR = ROOT / "runs" / "engine" / "g4-galaxy-prediction-contract-transform-registration.json"
ARTIFACT = ROOT / "runs" / "engine" / "g4-galaxy-manifest-bundle-tooling-readiness.json"
SOURCE = ROOT / "src" / "sigma_theory_compiler" / "g4_galaxy_manifest_bundle_tooling.py"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registration() -> dict[str, str | None]:
    predecessor = _load(PREDECESSOR)
    registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    registration.update(predecessor["preserved_predecessor_registration_fields"])
    registration.update(predecessor["newly_filled_registration_fields"])
    return registration


def _synthetic_audit_and_registration() -> tuple[dict, dict[str, str | None]]:
    registration = _registration()
    transform = registration["raw_to_calibrated_transform_sha256"]
    assert isinstance(transform, str)
    audit = audit_dataset_manifest(_synthetic_manifest(transform))
    registration.update(
        {
            "dataset_manifest_independent_audit_sha256": audit["content_sha256"],
            "galaxy_split_commitment_sha256": audit["galaxy_split_commitment_sha256"],
            "selected_primary_calibration_root_sha256": audit[
                "selected_primary_calibration_root_sha256"
            ],
            "selected_primary_imaging_and_spectroscopy_root_sha256": audit[
                "selected_primary_imaging_and_spectroscopy_root_sha256"
            ],
            "training_only_checkpoint_sha256": audit["training_only_checkpoint_sha256"],
        }
    )
    return audit, registration


def _rehash_manifest(manifest: dict) -> None:
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    manifest["content_sha256"] = _sha(body)


def test_tooling_contract_is_exact_and_readiness_is_disabled() -> None:
    contract = _load(CONTRACT)
    validate_tooling_contract(contract)
    body = {key: value for key, value in contract.items() if key != "content_sha256"}
    assert _sha(body) == contract["content_sha256"] == TOOLING_CONTRACT_SHA256
    assert set(contract["checked_in_readiness"].values()) == {False}
    assert (
        contract["prediction_bundle_builder_contract"]["synthetic_drafts_registration_admissible"]
        is False
    )
    assert contract["data_eligibility"] == ELIGIBILITY


def test_synthetic_manifest_audit_checks_roots_groups_and_remains_nonregistrable() -> None:
    registration = _registration()
    transform = registration["raw_to_calibrated_transform_sha256"]
    assert isinstance(transform, str)
    manifest = _synthetic_manifest(transform)
    audit = audit_dataset_manifest(manifest)
    assert audit["entry_count"] == 3
    assert audit["whole_galaxy_group_count"] == 2
    assert audit["group_leakage_found"] is False
    assert audit["target_values_opened"] is False
    assert audit["registration_admissible"] is False
    assert audit["registry_receipt_content_sha256"] is None


@pytest.mark.parametrize("mutation", ["halo", "redshift", "object_parameter", "target"])
def test_manifest_halo_redshift_object_parameter_and_target_leaks_reject(
    mutation: str,
) -> None:
    transform = _registration()["raw_to_calibrated_transform_sha256"]
    assert isinstance(transform, str)
    manifest = _synthetic_manifest(transform)
    if mutation == "halo":
        manifest["dark_matter_or_halo_inputs"] = True
    elif mutation == "redshift":
        manifest["redshift_distance_inputs"] = True
    elif mutation == "object_parameter":
        manifest["object_specific_gravity_parameters"] = {"gravity_scale": 2.0}
    else:
        entry = manifest["entries"][0]
        entry["target_values_opened"] = True
        entry["content_sha256"] = _sha(
            {key: value for key, value in entry.items() if key != "content_sha256"}
        )
    _rehash_manifest(manifest)
    with pytest.raises(ValueError, match="sealed schema|leaks"):
        audit_dataset_manifest(manifest)


def test_whole_galaxy_role_leak_and_root_tamper_reject() -> None:
    transform = _registration()["raw_to_calibrated_transform_sha256"]
    assert isinstance(transform, str)
    manifest = _synthetic_manifest(transform)
    entry = manifest["entries"][1]
    entry["split_role"] = "formula_selection_validation"
    entry["content_sha256"] = _sha(
        {key: value for key, value in entry.items() if key != "content_sha256"}
    )
    _rehash_manifest(manifest)
    with pytest.raises(ValueError, match="split leakage"):
        audit_dataset_manifest(manifest)
    manifest = _synthetic_manifest(transform)
    manifest["selected_primary_calibration_root_sha256"] = "0" * 64
    _rehash_manifest(manifest)
    with pytest.raises(ValueError, match="root or split commitment"):
        audit_dataset_manifest(manifest)


def test_nonindependent_or_unbound_registry_receipt_rejects() -> None:
    transform = _registration()["raw_to_calibrated_transform_sha256"]
    assert isinstance(transform, str)
    manifest = _synthetic_manifest(transform)
    receipt_body = {
        "schema_version": RECEIPT_SCHEMA,
        "manifest_content_sha256": manifest["content_sha256"],
        "source_registry_root_sha256": "a" * 64,
        "independent_reviewer_identity_sha256": "b" * 64,
        "reviewer_is_generator_operator": True,
        "observational_authorization": False,
    }
    receipt = {**receipt_body, "content_sha256": _sha(receipt_body)}
    with pytest.raises(ValueError, match="receipt is invalid"):
        audit_dataset_manifest(manifest, receipt)


def test_synthetic_bundle_draft_is_deterministic_valid_and_nonregistrable() -> None:
    audit, registration = _synthetic_audit_and_registration()
    first = build_prediction_bundle_draft(registration, audit, synthetic_control=True)
    second = build_prediction_bundle_draft(registration, audit, synthetic_control=True)
    assert first == second
    validate_prediction_bundle_draft(first, registration)
    assert first["registration_admissible"] is False
    assert first["synthetic_control"] is True
    assert first["observational_data_opened"] is False
    assert first["bundle"]["object_specific_gravity_parameter_count"] == 0
    assert first["prediction_bundle_content_sha256"] == first["prediction_bundle_file_sha256"]
    with pytest.raises(ValueError, match="independent registry receipt"):
        build_prediction_bundle_draft(registration, audit, synthetic_control=False)


def test_bundle_lineage_object_parameter_and_hash_tampering_reject() -> None:
    audit, registration = _synthetic_audit_and_registration()
    wrong = dict(registration)
    wrong["branch_and_domain_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate contract lineage"):
        build_prediction_bundle_draft(wrong, audit, synthetic_control=True)
    draft = build_prediction_bundle_draft(registration, audit, synthetic_control=True)
    mutated = copy.deepcopy(draft)
    mutated["bundle"]["object_specific_gravity_parameter_count"] = 1
    mutated["prediction_bundle_content_sha256"] = _sha(mutated["bundle"])
    mutated["prediction_bundle_file_sha256"] = _sha(mutated["bundle"])
    with pytest.raises(ValueError, match="exact bundle contract"):
        validate_prediction_bundle_draft(mutated, registration)


def test_artifact_rebuilds_without_advancing_exact_eleven_seven_ledger() -> None:
    stored = _load(ARTIFACT)
    assert build_g4_galaxy_manifest_bundle_tooling_readiness(_load(CONFIG), ROOT) == stored
    assert stored["content_sha256"] == (
        "902dbc9475f9eca2454c244f9de0f844f42182fefc4da5ea4096049bc94e6a4d"
    )
    assert _file_sha(ARTIFACT) == (
        "aa1722821d116ff60348dde324744b605dbf01dbc91b9c0eaa3523f59985fcf6"
    )
    assert stored["newly_filled_registration_fields"] == {}
    assert stored["filled_registration_hash_count"] == 11
    assert stored["missing_registration_hash_count"] == 7
    assert len(stored["unfilled_registration_fields"]) == 7
    assert stored["tooling_readiness"]["enabled"] is False
    assert stored["tooling_readiness"]["registration_fields_filled"] == 0
    assert stored["synthetic_controls"]["synthetic_values_promoted"] is False
    assert stored["prediction_bundle_registered"] is False


def test_bindings_authorization_provenance_and_seals_are_fail_closed() -> None:
    config = _load(CONFIG)
    assert config["source_bindings"]["tooling_source"]["file_sha256"] == _file_sha(SOURCE)
    for key in ("predecessor", "tooling_contract", "prediction_bundle_contract"):
        mutated = _load(CONFIG)
        mutated["source_bindings"][key]["content_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="content changed"):
            build_g4_galaxy_manifest_bundle_tooling_readiness(mutated, ROOT)
    mutated = _load(CONFIG)
    mutated["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened observations"):
        build_g4_galaxy_manifest_bundle_tooling_readiness(mutated, ROOT)
    artifact = _load(ARTIFACT)
    provenance = artifact["provenance"]
    body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert _sha(body) == provenance["binding_sha256"]
    assert artifact["observational_data_opened"] is False
    assert artifact["primary_record_access_count"] == 0
    assert artifact["dark_matter_or_halo_inputs"] is False
    assert artifact["redshift_distance_inputs"] is False
    assert artifact["object_specific_gravity_parameter_count"] == 0
    assert artifact["paid_llm_spend_usd"] == 0.0
