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
        "metadata_sources": 6,
        "candidate_lanes": 4,
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


def test_all_source_lanes_record_specific_unresolved_requirements() -> None:
    receipt = contract.build_receipt(ROOT)
    assert [row["lane_id"] for row in receipt["source_audit"]] == list(contract.LANE_IDS)
    assert all("BLOCKED" in row["decision"] for row in receipt["source_audit"])
    assert all(row["blocking_readiness_fields"] for row in receipt["source_audit"])


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
