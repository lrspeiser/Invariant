from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_constraint_boundary_embedding_audit import (
    build_future_aether_constraint_boundary_embedding_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_constraint_boundary_embedding_audit.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-constraint-boundary-embedding-audit.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-candidate-formal-followup.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_constraint_boundary_embedding_audit(_config(), ROOT)


def test_exact_blocked_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {
        "constraint_satisfying_asymptotically_Euclidean_completion_of_negative_twist_witness": 14
    }
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0
    assert rebuilt["constraint_satisfying_negative_total_energy_datum_count"] == 0


def test_every_source_record_and_action_binding_is_preserved(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    assert len(expected) == 14
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert (
            item["action_density_equivalence_sha256"]
            == predecessor["action_density_equivalence_sha256"]
        )
        assert item["compilation_receipt_sha256"] == predecessor["compilation_receipt_sha256"]
        assert item["source_followup_record_sha256"] == predecessor["content_sha256"]
        assert item["exact_specialization"] == predecessor["exact_specialization"]


def test_affine_completion_fails_both_constraints_and_AE_boundary(rebuilt: dict) -> None:
    assert rebuilt["explicit_affine_ansatz_constraint_reject_count"] == 14
    assert rebuilt["nonzero_Hamiltonian_constraint_residual_count"] == 14
    assert rebuilt["nonzero_momentum_constraint_residual_count"] == 14
    assert rebuilt["undefined_AE_boundary_contribution_count"] == 14
    assert rebuilt["Hamiltonian_constraint_residual_counts"] == {
        "-283/288": 2,
        "-31/96": 4,
        "-47/128": 4,
        "-71/128": 4,
    }
    assert rebuilt["momentum_constraint_residual_norm_squared_counts"] == {
        "1369/13824": 4,
        "1849/32768": 4,
        "3249/32768": 4,
        "3481/2592": 2,
    }
    for item in rebuilt["candidate_records"]:
        certificate = item["affine_constraint_boundary_certificate"]
        assert Fraction(certificate["normalized_local_twist_hamiltonian_H_core"]) < 0
        assert Fraction(certificate["flat_static_Hamiltonian_constraint_residual"]) != 0
        assert Fraction(certificate["flat_static_momentum_constraint_residual_norm_squared"]) > 0
        assert certificate["explicit_affine_ansatz_constraint_datum_status"] == "reject"
        assert certificate["asymptotically_Euclidean"] is False
        assert certificate["completed_Aether_boundary_energy"] == (
            "undefined_outside_AE_phase_space"
        )


def test_rejected_ansatz_never_becomes_a_rejected_candidate(rebuilt: dict) -> None:
    for item in rebuilt["candidate_records"]:
        gates = item["gate_ledger"]
        assert gates["flat_static_Hamiltonian_constraint"]["status"] == ("reject_explicit_ansatz")
        assert gates["flat_static_momentum_constraint"]["status"] == ("reject_explicit_ansatz")
        assert gates["asymptotically_Euclidean_boundary_completion"]["status"] == ("blocked")
        assert item["decision"] == "blocked"
        assert item["candidate_rejection_authorized"] is False
        assert item["constraint_satisfying_negative_total_energy_datum_proven"] is False
        assert item["formal_pass"] is False


def test_control_scope_hashes_provenance_and_seals(rebuilt: dict) -> None:
    assert rebuilt["reviewed_control_replay_count"] == 4
    assert (
        rebuilt["reviewed_control_evidence"]["einstein_aether_restricted_nonlinear_total_energy"][
            "applicability"
        ]
        == "boundary_identity_and_twisting_scope_exclusion_only"
    )
    assert rebuilt["full_candidate_specific_formal_completion_claimed"] is False
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == _config()["data_eligibility"]
    provenance = rebuilt["provenance"]
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)
    for item in rebuilt["candidate_records"]:
        body = {key: value for key, value in item.items() if key != "content_sha256"}
        assert item["content_sha256"] == _sha(body)
        record_provenance = item["provenance"]
        record_body = {
            key: value for key, value in record_provenance.items() if key != "binding_sha256"
        }
        assert record_provenance["binding_sha256"] == _sha(record_body)
        assert item["observational_data_opened"] is False
        assert item["solar_bundle_generated"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.update(
                data_eligibility={
                    **config["data_eligibility"],
                    "observational_data_opened": True,
                }
            ),
            "eligibility is open",
        ),
        (
            lambda config: config.update(observational_authorization=True),
            "opened observations",
        ),
        (
            lambda config: config.update(external_paid_llm_calls=True),
            "enabled paid LLM calls",
        ),
        (
            lambda config: config["reviewed_controls"].pop(),
            "control registry is incomplete",
        ),
        (
            lambda config: config["reviewed_controls"][0].update(
                applicability="full_candidate_formal_pass"
            ),
            "control registry is incomplete",
        ),
        (
            lambda config: config["source_followup_artifact"].update(content_sha256="0" * 64),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "file hash mismatch",
        ),
    ],
)
def test_open_seals_missing_controls_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_future_aether_constraint_boundary_embedding_audit(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_followup_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_constraint_boundary_embedding_audit(config, ROOT)
