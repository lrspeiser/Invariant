from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY
from sigma_theory_compiler.reviewed_g4_candidate_galaxy_evaluator import (
    ACTION_SHA256,
    BUNDLE_SCHEMA,
    CANDIDATE_ID,
    DESCRIPTOR_FIELD,
    FORMAL_PROVENANCE_SHA256,
    INPUT_CONTRACT,
    OUTPUT_CONTRACT,
    PREDICTION_BUNDLE_CONTRACT_SHA256,
    REQUIRED_REGISTRATION_HASHES,
    build_reviewed_g4_galaxy_evaluator_readiness,
    reviewed_g4_candidate_galaxy_evaluator,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reviewed_g4_candidate_galaxy_evaluator_readiness.json"
DESCRIPTOR = ROOT / "configs" / "reviewed_g4_candidate_galaxy_evaluator.json"
CONTRACT = ROOT / "configs" / "reviewed_g4_galaxy_prediction_bundle_contract.json"
ARTIFACT = (
    ROOT / "runs" / "engine" / "reviewed-g4-candidate-galaxy-evaluator-readiness.json"
)
SOURCE = (
    ROOT
    / "src"
    / "sigma_theory_compiler"
    / "reviewed_g4_candidate_galaxy_evaluator.py"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict:
    return {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "role": "generated_candidate",
        "data_eligibility": dict(ELIGIBILITY),
    }


def _future_context() -> dict:
    registration = {name: "1" * 64 for name in REQUIRED_REGISTRATION_HASHES}
    registration["prediction_bundle_contract_sha256"] = (
        PREDICTION_BUNDLE_CONTRACT_SHA256
    )
    registration[DESCRIPTOR_FIELD] = "2" * 64
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "formal_provenance_sha256": FORMAL_PROVENANCE_SHA256,
        "branch_and_domain_contract_sha256": registration[
            "branch_and_domain_contract_sha256"
        ],
        "input_contract": INPUT_CONTRACT,
        "output_contract": OUTPUT_CONTRACT,
        "universal_parameter_count": 0,
        "object_specific_gravity_parameter_count": 0,
        "rotation_prediction_implementation_sha256": registration[
            "rotation_prediction_implementation_sha256"
        ],
        "lensing_prediction_implementation_sha256": registration[
            "lensing_prediction_implementation_sha256"
        ],
        "baryonic_calibration_hierarchy_sha256": registration[
            "baryonic_calibration_hierarchy_sha256"
        ],
        "joint_covariance_contract_sha256": registration[
            "joint_covariance_contract_sha256"
        ],
        "likelihood_contract_sha256": registration["likelihood_contract_sha256"],
        "galaxy_split_commitment_sha256": registration[
            "galaxy_split_commitment_sha256"
        ],
        "training_only_checkpoint_sha256": registration[
            "training_only_checkpoint_sha256"
        ],
        "stopping_rule_sha256": registration["stopping_rule_sha256"],
        "distance_mode_contract_sha256": registration[
            "distance_mode_contract_sha256"
        ],
        "data_eligibility": dict(ELIGIBILITY),
        "observational_data_opened": False,
    }
    registration["prediction_bundle_content_sha256"] = hashlib.sha256(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "data_eligibility": dict(ELIGIBILITY),
        "observational_opening_authorized": False,
        "evaluator_descriptor_binding_sha256": registration[DESCRIPTOR_FIELD],
        "registration_hashes": registration,
        "prediction_bundle": bundle,
    }


def test_artifact_rebuilds_exactly_and_remains_sealed() -> None:
    stored = _load(ARTIFACT)
    rebuilt = build_reviewed_g4_galaxy_evaluator_readiness(_load(CONFIG), ROOT)
    assert rebuilt == stored
    assert stored["content_sha256"] == (
        "05d9cffd288746fdd297058fec61b21a96226bc169eef46892b53d1a08c3764c"
    )
    assert _file_sha(ARTIFACT) == (
        "29e066623914c84c88172fffb20e262d31635052cdef66188a709049aee6492f"
    )
    assert stored["decision"] == "blocked"
    assert stored["observational_authorization"] is False
    assert stored["observational_data_opened"] is False
    assert stored["primary_record_access_count"] == 0
    assert stored["dark_matter_or_halo_inputs"] is False
    assert stored["redshift_distance_inputs"] is False
    assert stored["tracking_target_values_opened"] is False
    assert stored["paid_llm_spend_usd"] == 0.0


def test_exact_action_formal_branch_and_ppn_lineage_are_bound() -> None:
    artifact = _load(ARTIFACT)
    assert artifact["candidate"]["candidate_id"] == CANDIDATE_ID
    assert artifact["candidate"]["action_sha256"] == ACTION_SHA256
    provenance = artifact["provenance"]
    assert provenance["formal_provenance_sha256"] == FORMAL_PROVENANCE_SHA256
    assert provenance["candidate_dossier_sha256"] == (
        "c3ce739cddfe70186096280c20602c2f47b9ad716898cdb480242f44b76cf010"
    )
    assert provenance["scalar_free_branch_sha256"] == (
        "2bca9d26343843231a8333bc9ac2396c395c388d24f55ae488c04c05f59256dc"
    )
    assert provenance["PPN_prediction_sha256"] == (
        "4e90877b2e49f682a6457e65f822c5b09d6773f5df041635cca70d1ecb8c12a2"
    )
    assert "not galaxy evidence" in artifact["interpretation"]


def test_descriptor_and_future_bundle_contract_are_exact_and_baryons_only() -> None:
    descriptor = _load(DESCRIPTOR)
    contract = _load(CONTRACT)
    assert descriptor["artifact_sha256"] == _file_sha(SOURCE)
    assert descriptor["prediction_bundle_contract_file_sha256"] == _file_sha(
        CONTRACT
    )
    assert (
        descriptor["prediction_bundle_contract_content_sha256"]
        == PREDICTION_BUNDLE_CONTRACT_SHA256
    )
    properties = contract["properties"]
    assert properties["candidate_id"]["const"] == CANDIDATE_ID
    assert properties["action_sha256"]["const"] == ACTION_SHA256
    assert properties["input_contract"]["const"] == INPUT_CONTRACT
    assert properties["output_contract"]["const"] == OUTPUT_CONTRACT
    assert properties["object_specific_gravity_parameter_count"]["const"] == 0
    assert "dark_matter_or_halo_label" in INPUT_CONTRACT["forbidden"]
    assert "redshift_derived_distance_or_environment" in INPUT_CONTRACT["forbidden"]
    assert OUTPUT_CONTRACT["lensing_formula_selection_use"] is False


def test_current_callback_fills_only_descriptor_and_lists_all_missing_hashes() -> None:
    artifact = _load(ARTIFACT)
    decision = artifact["current_evaluator_decision"]
    expected = sorted(set(REQUIRED_REGISTRATION_HASHES) - {DESCRIPTOR_FIELD})
    assert decision["decision"] == "blocked"
    assert decision["blocker"] == (
        "missing_registered_galaxy_prediction_and_data_contracts"
    )
    assert decision["filled_registration_hash_count"] == 1
    assert decision["missing_registration_hashes"] == expected
    assert artifact["unfilled_prediction_data_registration_fields"] == expected
    assert artifact["prediction_bundle_registered"] is False
    assert artifact["candidate_use_authorized"] is False


def test_synthetic_shape_and_covariance_controls_are_not_candidate_evidence() -> None:
    controls = _load(ARTIFACT)["synthetic_controls"]
    shape = controls["shape"]
    covariance = controls["covariance"]
    assert shape["decision"] == "pass"
    assert shape["role"] == "synthetic_shape_only_not_candidate_prediction"
    assert shape["object_specific_gravity_parameter_count"] == 0
    assert covariance["decision"] == "pass"
    assert covariance["role"] == "synthetic_covariance_only_not_candidate_evidence"
    assert covariance["joint_rotation_lensing_covariance_shape"] == [11, 11]
    assert covariance["positive_definite"] is True
    assert covariance["cross_channel_covariance_nonzero"] is True
    assert shape["observational_data_opened"] is False
    assert covariance["observational_data_opened"] is False


def test_empty_partial_and_fully_registered_contexts_all_fail_closed() -> None:
    base = {
        "data_eligibility": dict(ELIGIBILITY),
        "observational_opening_authorized": False,
    }
    empty = reviewed_g4_candidate_galaxy_evaluator(_candidate(), base)
    assert empty["decision"] == "blocked"
    assert len(empty["missing_registration_hashes"]) == len(
        REQUIRED_REGISTRATION_HASHES
    )

    partial_context = copy.deepcopy(base)
    partial_context["registration_hashes"] = {DESCRIPTOR_FIELD: "2" * 64}
    partial = reviewed_g4_candidate_galaxy_evaluator(
        _candidate(), partial_context
    )
    assert partial["decision"] == "blocked"
    assert partial["filled_registration_hash_count"] == 1
    assert "dataset_manifest_independent_audit_sha256" in partial[
        "missing_registration_hashes"
    ]

    full = reviewed_g4_candidate_galaxy_evaluator(_candidate(), _future_context())
    assert full["decision"] == "blocked"
    assert full["blocker"] == "separate_observational_opening_authorization_required"
    assert full["readiness"] == "fully_registered_prediction_bundle_validated"
    assert full["missing_registration_hashes"] == []
    assert full["observational_data_opened"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("candidate", "candidate identity changed"),
        ("authorization", "cannot authorize data opening"),
        ("eligibility", "context eligibility changed"),
        ("object_parameter", "violates the sealed contract"),
        ("halo_input", "violates the sealed contract"),
        ("contract_hash", "contract or evaluator descriptor binding changed"),
        ("descriptor_binding", "contract or evaluator descriptor binding changed"),
        ("bundle_hash", "future galaxy bundle hash mismatch"),
    ],
)
def test_identity_policy_contract_and_bundle_tampering_is_rejected(
    mutation: str, message: str
) -> None:
    candidate = _candidate()
    context = _future_context()
    if mutation == "candidate":
        candidate["action_sha256"] = "0" * 64
    elif mutation == "authorization":
        context["observational_opening_authorized"] = True
    elif mutation == "eligibility":
        context["data_eligibility"] = {**ELIGIBILITY, "redshift_distance_inputs": True}
    elif mutation == "object_parameter":
        context["prediction_bundle"]["object_specific_gravity_parameter_count"] = 1
    elif mutation == "halo_input":
        context["prediction_bundle"]["input_contract"] = {
            **INPUT_CONTRACT,
            "channels": [*INPUT_CONTRACT["channels"], "dark_matter_halo_mass"],
        }
    elif mutation == "contract_hash":
        context["registration_hashes"]["prediction_bundle_contract_sha256"] = (
            "3" * 64
        )
    elif mutation == "descriptor_binding":
        context["evaluator_descriptor_binding_sha256"] = "3" * 64
    else:
        context["prediction_bundle"]["lensing_prediction_implementation_sha256"] = (
            "4" * 64
        )
    with pytest.raises(ValueError, match=message):
        reviewed_g4_candidate_galaxy_evaluator(candidate, context)


def test_source_binding_and_observational_config_tampering_is_rejected() -> None:
    config = _load(CONFIG)
    config["source_bindings"]["candidate_dossier"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact changed"):
        build_reviewed_g4_galaxy_evaluator_readiness(config, ROOT)

    config = _load(CONFIG)
    config["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened observations"):
        build_reviewed_g4_galaxy_evaluator_readiness(config, ROOT)
