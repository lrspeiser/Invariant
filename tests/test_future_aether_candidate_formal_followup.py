from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_candidate_formal_followup import (
    build_future_aether_candidate_formal_followup,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_candidate_formal_followup.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-candidate-formal-followup.json"
PREFLIGHT_PATH = ROOT / "runs/engine/reviewed-future-parameter-formal-preflight-001.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_candidate_formal_followup(_config(), ROOT)


def test_exact_14_candidate_blocked_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["input_preflight_pass_count"] == rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {
        "full_constraint_embedding_of_negative_static_twist_jet": 14
    }
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0
    assert rebuilt["exact_negative_local_twist_witness_count"] == 14


def test_every_target_is_rederived_from_an_exact_preflight_pass(rebuilt: dict) -> None:
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    expected = {
        item["candidate_id"]: item
        for item in preflight["candidate_records"]
        if item["family_id"] == "AETHER_K1234_PARAMETER_CELL" and item["decision"] == "pass"
    }
    assert len(expected) == 14
    assert {item["candidate_id"] for item in rebuilt["candidate_records"]} == set(expected)
    for item in rebuilt["candidate_records"]:
        source = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == source["typed_action_ir_sha256"]
        assert item["compilation_receipt_sha256"] == source["compilation_receipt_sha256"]
        assert item["source_preflight_record_sha256"] == source["content_sha256"]
        assert item["parameter_cell_lineage_sha256"] == source["parameter_cell_lineage_sha256"]
        assert item["exact_specialization"] == source["exact_specialization"]


def test_exact_local_twist_witnesses_block_but_do_not_reject(rebuilt: dict) -> None:
    assert rebuilt["witness_tilt_squared_counts"] == {"1": 8, "2": 4, "8": 2}
    for item in rebuilt["candidate_records"]:
        witness = item["exact_specialization"]["finite_negative_twist_witness"]
        assert Fraction(witness["C_y"]) < 0
        assert witness["local_hamiltonian_density_negative"] is True
        assert witness["full_gravitational_constraint_embedding_proven"] is False
        assert witness["candidate_rejection_authorized_by_this_witness_alone"] is False
        assert item["decision"] == "blocked"
        assert item["candidate_rejection_authorized"] is False
        assert item["formal_pass"] is False
        local = item["gate_ledger"]["static_unit_reduced_pure_twist_local_energy"]
        assert local["status"] == "blocked"
        assert "constraint surface" in local["reason"]


def test_adapter_scope_and_tilt_strata_are_not_overclaimed(rebuilt: dict) -> None:
    assert rebuilt["reviewed_adapter_replay_count"] == 6
    assert rebuilt["reviewed_bound_cadabra_control_count"] == 1
    assert rebuilt["reviewed_arbitrary_background_noether_control"]["status"] == "pass"
    assert rebuilt["global_tilt_strata_counts"] == {
        "finite_characteristic_foliation_present": 13,
        "globally_noncharacteristic_for_finite_unit_tilt": 1,
    }
    adapters = rebuilt["reviewed_adapter_evidence"]
    assert adapters["maxwell_unit_aether_nonlinear_hamiltonian"]["applicability"] == (
        "control_only_not_action_equivalent"
    )
    assert (
        adapters["einstein_aether_restricted_nonlinear_total_energy"]["applicability"]
        == "restricted_subsector_scope_exclusion_only"
    )
    for item in rebuilt["candidate_records"]:
        gates = item["gate_ledger"]
        assert gates["arbitrary_background_covariant_Noether_identity"]["status"] == "pass"
        assert gates["maxwell_unit_aether_control"]["status"] == (
            "not_applicable_different_action_subclass"
        )
        assert gates["restricted_nonlinear_positive_energy_theorem"]["status"] == (
            "not_applicable_to_generic_twisting_sector"
        )
        assert gates["generic_twisting_constraint_reduced_hamiltonian"]["status"] == ("blocked")
        assert gates["global_positive_energy"]["status"] == "blocked"


def test_hash_provenance_formal_scope_and_data_seals(rebuilt: dict) -> None:
    assert rebuilt["candidate_specific_formal_followup_completed"] is True
    assert rebuilt["full_candidate_specific_formal_completion_claimed"] is False
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["solar_bundle_count"] == 0
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
            lambda config: config["reviewed_adapters"].pop(),
            "adapter registry is incomplete",
        ),
        (
            lambda config: config["reviewed_adapters"][0].update(
                applicability="full_candidate_formal_pass"
            ),
            "adapter registry is incomplete",
        ),
        (
            lambda config: config["source_preflight_artifact"].update(content_sha256="0" * 64),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "implementation file hash mismatch",
        ),
    ],
)
def test_open_seals_missing_adapters_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_future_aether_candidate_formal_followup(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_preflight_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_candidate_formal_followup(config, ROOT)
