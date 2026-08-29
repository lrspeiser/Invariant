from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_independent_data_contract as contract

ROOT = Path(__file__).resolve().parents[1]


def test_source_audit_is_complete_but_selection_and_targets_stay_sealed() -> None:
    receipt = contract.build_receipt(ROOT)
    assert receipt["decision"] == "SOURCE_AUDIT_COMPLETE_SELECTION_BLOCKED_TARGETS_SEALED"
    assert receipt["counts"] == {
        "metadata_sources": 11,
        "candidate_lanes": 6,
        "fully_ready_lanes": 0,
        "selected_lanes": 0,
        "payloads_opened": 0,
        "target_rows_opened": 0,
        "transformations": 5,
        "manifest_required_fields": 18,
    }
    assert receipt["claims"] == {
        "source_metadata_audit_complete": True,
        "independent_source_selected": False,
        "independent_data_ready": False,
        "observational_authorization": False,
        "payload_accessed": False,
        "target_rows_accessed": False,
        "scientific_result_emitted": False,
    }
    assert set(receipt["completed_goal_evidence"]) == {"CP3.7", "CP3.8", "CP7.1"}
    assert receipt["gate_status"] == {"CP3": "PARTIAL", "CP7": "PARTIAL"}
    assert {"CP7.2", "CP7.3", "CP7.9"} <= set(receipt["blocked_goal_evidence"])


def test_all_source_lanes_record_specific_unresolved_requirements() -> None:
    receipt = contract.build_receipt(ROOT)
    assert [row["lane_id"] for row in receipt["source_audit"]] == list(contract.LANE_IDS)
    assert all("BLOCKED" in row["decision"] for row in receipt["source_audit"])
    assert all(row["blocking_readiness_fields"] for row in receipt["source_audit"])
    for row in receipt["source_audit"]:
        assert row["audit_details"]["exact_missing_fields"] == row[
            "blocking_readiness_fields"
        ]
        assert row["audit_details"]["payload_commitment"] is None


def test_new_lanes_are_metadata_only_and_not_population_or_covariance_ready() -> None:
    config = contract.load_config(ROOT)
    lanes = {lane["lane_id"]: lane for lane in config["candidate_lanes"]}

    a302 = lanes["CHEX_MATE_A302_PRESSURE_SUBSAMPLE"]
    assert "28" in a302["sample_scope"]
    assert "24" in a302["sample_scope"]
    assert not a302["readiness"]["stellar_baryon_profile_files_verified"]
    assert "MILCA y-map is non-public" in a302["audit_details"]["covariance_blocker"]
    assert "120-object minimum" in a302["audit_details"]["population_and_power_limitations"]

    act_erass = lanes["ACT_DR6_ERASS1_PROSPECTIVE_REDUCTION"]
    assert "10,040" in act_erass["sample_scope"]
    assert "12,247" in act_erass["sample_scope"]
    assert not act_erass["readiness"]["full_covariance_files_verified"]
    assert "unknown" in act_erass["audit_details"]["population_and_power_limitations"]
    assert act_erass["audit_details"]["payload_commitment"] is None


def test_redshift_formula_leakage_and_missing_packet_roles_fail_closed() -> None:
    config = contract.load_config(ROOT)
    leaked = copy.deepcopy(config)
    leaked["cosmology_and_redshift_freeze"]["prohibited_redshift_uses"].remove(
        "candidate_formula_input"
    )
    with pytest.raises(contract.GravityClusterDataContractError, match="redshift"):
        contract.validate_config(leaked)

    weakened = copy.deepcopy(config)
    weakened["role_requirements"]["covariance_roles"].pop()
    with pytest.raises(contract.GravityClusterDataContractError, match="roles"):
        contract.validate_config(weakened)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["selection_state"].__setitem__(
            "selected_primary_lane", "CHEX_MATE_XMM_THERMODYNAMICS"
        ),
        lambda value: value["selection_state"].__setitem__(
            "observational_authorization", True
        ),
        lambda value: value["selection_state"].__setitem__(
            "independent_target_rows_opened", 1
        ),
        lambda value: value["seals"].__setitem__("payload_accessed", True),
        lambda value: value["candidate_lanes"][0].__setitem__("payload_opened", True),
        lambda value: value["candidate_lanes"][4]["audit_details"].__setitem__(
            "payload_commitment", "unfrozen-file-set"
        ),
        lambda value: value["candidate_lanes"][5]["audit_details"][
            "exact_missing_fields"
        ].pop(),
    ],
)
def test_selection_authorization_and_payload_mutations_fail_closed(mutation: object) -> None:
    config = copy.deepcopy(contract.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(contract.GravityClusterDataContractError):
        contract.validate_config(config)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / contract.OUTPUT_PATH).read_text(encoding="utf-8"))
    contract.validate_receipt(stored, ROOT)
    assert stored == contract.build_receipt(ROOT)
